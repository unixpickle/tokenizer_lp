from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    all_word_token_colors,
    build_standard_form,
    chunked,
    count_pretokenized_strings,
    enumerate_word_edge_paths_by_pattern,
    pair_hull_current_violation,
    pair_reduced_fractional_edge_hull_cut_from_paths,
    prepare_lp_data,
    projected_pair_hull_path_signatures,
    rank_short_fractional_words,
    short_word_pair_candidates,
    tupleify_json_key,
)
from tokenisation_lp.pretokenization import build_pretokenizer


LOGGER = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    run_dir = Path(args.run_dir).expanduser()
    state_dir = run_dir / "lp" / "training_state"
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading checkpoint from %s", state_dir)
    checkpoint = load_checkpoint(state_dir)
    existing_cut_keys = checkpoint["existing_cut_keys"]
    existing_pair_prefixes = {
        key[:3]
        for key in existing_cut_keys
        if len(key) >= 3 and key[0] == "short_word_pair_hull"
    }
    x_values = np.load(state_dir / "latest_solution.npy", allow_pickle=False)

    LOGGER.info("Loading corpus and rebuilding LP")
    texts = load_texts(args.data_dir)
    pretokenizer, _ = build_pretokenizer(args.pretokenizer)
    word_counts = count_pretokenized_strings(texts, pretokenizer)
    words = list(word_counts)
    freqs = [word_counts[word] for word in words]
    edges, free_edges, num_vertices, tokens = prepare_lp_data(
        words,
        freqs,
        min_token_count=args.min_token_count,
        max_token_length=args.max_token_length,
    )
    lp = build_standard_form(edges, freqs, tokens, free_edges, num_vertices)
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    expected = num_f + num_g + lp["num_tokens"]
    if x_values.shape != (expected,):
        raise ValueError(f"latest_solution shape {x_values.shape} does not match LP vars {expected}")
    f_values = x_values[:num_f]
    g_values = x_values[num_f : num_f + num_g]
    t_values = x_values[num_f + num_g :]

    prep_start = time.monotonic()
    fractional_colors = set(np.flatnonzero((t_values > args.cut_tolerance) & (t_values < 1.0 - args.cut_tolerance)))
    word_colors = {}
    path_pattern_cache = {}
    paths_by_word = {}
    pair_tasks, pair_prep = prepare_pair_tasks(
        args,
        lp,
        f_values,
        t_values,
        fractional_colors,
        existing_pair_prefixes,
        word_colors,
        paths_by_word,
        path_pattern_cache,
    )
    triple_tasks, triple_prep = prepare_triple_tasks(
        args,
        lp,
        f_values,
        t_values,
        fractional_colors,
        word_colors,
        paths_by_word,
        path_pattern_cache,
    )
    prep_seconds = time.monotonic() - prep_start
    LOGGER.info(
        "Prepared frontier tasks: pairs=%d triples=%d prep=%.1fs paths=%d patterns=%d",
        len(pair_tasks),
        len(triple_tasks),
        prep_seconds,
        len(paths_by_word),
        len(path_pattern_cache),
    )

    worker_state = {
        "lp": lp,
        "f_values": f_values,
        "g_values": g_values,
        "t_values": t_values,
        "paths_by_word": paths_by_word,
        "tolerance": args.cut_tolerance,
        "pair_row_limit": args.pair_row_limit,
        "triple_row_limit": args.triple_row_limit,
    }
    pair_records = run_tasks(
        pair_tasks,
        worker_state,
        pair_batch_worker,
        init_worker,
        args.workers,
        args.batch_size,
        "pair",
        args.progress_interval,
    )
    triple_records = run_tasks(
        triple_tasks,
        worker_state,
        triple_batch_worker,
        init_worker,
        args.workers,
        args.batch_size,
        "triple",
        args.progress_interval,
    )

    pair_cuts_path = output_dir / "pair_cuts.jsonl"
    triple_cuts_path = output_dir / "triple_cuts.jsonl"
    write_cut_records(pair_cuts_path, pair_records, words)
    write_cut_records(triple_cuts_path, triple_records, words)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_next_iteration": checkpoint.get("next_iteration"),
        "active_cuts": len(existing_cut_keys),
        "existing_pair_prefixes": len(existing_pair_prefixes),
        "fractional_colors": len(fractional_colors),
        "prep_seconds": prep_seconds,
        "pair_prep": pair_prep,
        "triple_prep": triple_prep,
        "pairs": summarize_records(pair_records, pair_prep.get("remaining_candidate_pairs")),
        "triples": summarize_records(triple_records, triple_prep.get("estimated_candidate_triples")),
        "pair_cuts_path": str(pair_cuts_path),
        "triple_cuts_path": str(triple_cuts_path),
    }
    summary_path = output_dir / "summary.json"
    log_path = output_dir / "summary.log"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = format_summary(summary)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        LOGGER.info(line)
    LOGGER.info("Wrote summary=%s pair_cuts=%s triple_cuts=%s", summary_path, pair_cuts_path, triple_cuts_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore remaining pair-cut and triplet-cut frontier from a saved LP checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8)
    parser.add_argument("--cut-tolerance", type=float, default=1e-6)
    parser.add_argument("--pair-max-words", type=int, default=700)
    parser.add_argument("--pair-max-word-length", type=int, default=12)
    parser.add_argument("--pair-top-words-per-color", type=int, default=48)
    parser.add_argument("--candidate-word-multiplier", type=float, default=16.0)
    parser.add_argument("--candidate-top-words-multiplier", type=float, default=16.0)
    parser.add_argument("--pair-min-fractional-shared-colors", type=int, default=2)
    parser.add_argument("--pair-sample", type=int, default=80000)
    parser.add_argument("--pair-row-limit", type=int, default=200000)
    parser.add_argument("--triple-max-words", type=int, default=700)
    parser.add_argument("--triple-top-words-per-color", type=int, default=48)
    parser.add_argument("--triple-token-mode", choices=("shared_all", "at_least_two"), default="at_least_two")
    parser.add_argument("--triple-min-fractional-colors", type=int, default=2)
    parser.add_argument("--triple-candidate-sample", type=int, default=250000)
    parser.add_argument("--triple-sample", type=int, default=30000)
    parser.add_argument("--triple-row-limit", type=int, default=200000)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=5000)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def load_checkpoint(state_dir: Path) -> dict:
    payload = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    payload["existing_cut_keys"] = {
        tupleify_json_key(key)
        for key in payload.get("existing_cut_keys", [])
    }
    return payload


def prepare_pair_tasks(args, lp, f_values, t_values, fractional_colors, existing_pair_prefixes, word_colors, paths_by_word, path_pattern_cache):
    rng = random.Random(args.random_seed)
    candidate_max_words = max(args.pair_max_words, int(math.ceil(args.pair_max_words * max(1.0, args.candidate_word_multiplier))))
    candidate_top_words = max(
        args.pair_top_words_per_color,
        int(math.ceil(args.pair_top_words_per_color * max(1.0, args.candidate_top_words_multiplier))),
    )
    pair_rows, _ = short_word_pair_candidates(
        lp,
        f_values,
        t_values,
        max_words=candidate_max_words,
        max_word_length=args.pair_max_word_length,
        top_words_per_color=candidate_top_words,
        tolerance=args.cut_tolerance,
    )
    rng.shuffle(pair_rows)
    tasks = []
    skipped = Counter()
    remaining_candidate_pairs = 0
    for _, left_word, right_word in pair_rows:
        if ("short_word_pair_hull", left_word, right_word) in existing_pair_prefixes:
            skipped["existing_cut"] += 1
            continue
        remaining_candidate_pairs += 1
        if len(tasks) >= args.pair_sample:
            continue
        left_colors = word_colors.setdefault(left_word, set(all_word_token_colors(lp, left_word)))
        right_colors = word_colors.setdefault(right_word, set(all_word_token_colors(lp, right_word)))
        selected_tokens = tuple(sorted((left_colors & right_colors) & fractional_colors))
        if len(selected_tokens) < args.pair_min_fractional_shared_colors:
            skipped["shared"] += 1
            continue
        ensure_paths(lp, left_word, paths_by_word, path_pattern_cache, args.max_paths)
        ensure_paths(lp, right_word, paths_by_word, path_pattern_cache, args.max_paths)
        if paths_by_word[left_word] is None or paths_by_word[right_word] is None:
            skipped["paths"] += 1
            continue
        tasks.append((left_word, right_word, selected_tokens))
    return tasks, {
        "candidate_pairs": len(pair_rows),
        "remaining_candidate_pairs": remaining_candidate_pairs,
        "sampled_tasks": len(tasks),
        "skipped": dict(skipped),
        "candidate_words": candidate_max_words,
        "candidate_top_words": candidate_top_words,
    }


def prepare_triple_tasks(args, lp, f_values, t_values, fractional_colors, word_colors, paths_by_word, path_pattern_cache):
    rng = random.Random(args.random_seed + 17)
    max_words = max(args.triple_max_words, int(math.ceil(args.triple_max_words * max(1.0, args.candidate_word_multiplier))))
    top_words = max(
        args.triple_top_words_per_color,
        int(math.ceil(args.triple_top_words_per_color * max(1.0, args.candidate_top_words_multiplier))),
    )
    ranked_words = rank_short_fractional_words(
        lp,
        f_values,
        t_values,
        max_words=max_words,
        max_word_length=args.pair_max_word_length,
        tolerance=args.cut_tolerance,
    )
    color_to_words = defaultdict(list)
    for word_idx in ranked_words:
        scores = defaultdict(float)
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            edge_value = float(f_values[edge_idx])
            if not (args.cut_tolerance < edge_value < 1.0 - args.cut_tolerance):
                continue
            info = lp["nonfree_edge_info"][edge_idx]
            token_idx = info["token_index"]
            token_value = float(t_values[token_idx])
            if token_idx in fractional_colors and args.cut_tolerance < token_value < 1.0 - args.cut_tolerance:
                scores[token_idx] += min(edge_value, 1.0 - edge_value) * max(1, info["end"] - info["start"])
        word_colors[word_idx] = set(all_word_token_colors(lp, word_idx))
        for token_idx, score in scores.items():
            color_to_words[token_idx].append((score, word_idx))
    for rows in color_to_words.values():
        rows.sort(reverse=True)

    triple_candidates = set()
    estimated_candidate_triples = 0
    colors = list(color_to_words)
    rng.shuffle(colors)
    per_color_budget = max(1, args.triple_candidate_sample // max(1, len(colors)))
    for token_idx in colors:
        words = [word_idx for _, word_idx in color_to_words[token_idx][:top_words]]
        if len(words) < 3:
            continue
        estimated_candidate_triples += math.comb(len(words), 3)
        if len(triple_candidates) >= args.triple_candidate_sample:
            continue
        for _ in range(per_color_budget):
            triple_candidates.add(tuple(sorted(rng.sample(words, 3))))
            if len(triple_candidates) >= args.triple_candidate_sample:
                break

    candidates = list(triple_candidates)
    rng.shuffle(candidates)
    tasks = []
    skipped = Counter()
    for words3 in candidates:
        if len(tasks) >= args.triple_sample:
            break
        color_sets = [word_colors.setdefault(word_idx, set(all_word_token_colors(lp, word_idx))) for word_idx in words3]
        if args.triple_token_mode == "shared_all":
            selected_set = set.intersection(*color_sets) & fractional_colors
        else:
            counts = Counter(token_idx for colors in color_sets for token_idx in colors if token_idx in fractional_colors)
            selected_set = {token_idx for token_idx, count in counts.items() if count >= 2}
        selected_tokens = tuple(sorted(selected_set))
        if len(selected_tokens) < args.triple_min_fractional_colors:
            skipped["shared"] += 1
            continue
        ok = True
        for word_idx in words3:
            ensure_paths(lp, word_idx, paths_by_word, path_pattern_cache, args.max_paths)
            if paths_by_word[word_idx] is None:
                ok = False
                break
        if not ok:
            skipped["paths"] += 1
            continue
        tasks.append((*words3, selected_tokens))
    return tasks, {
        "candidate_words": max_words,
        "candidate_top_words": top_words,
        "candidate_triples_sampled": len(candidates),
        "estimated_candidate_triples": estimated_candidate_triples,
        "sampled_tasks": len(tasks),
        "skipped": dict(skipped),
        "token_mode": args.triple_token_mode,
    }


def ensure_paths(lp, word_idx, paths_by_word, path_pattern_cache, max_paths):
    if word_idx not in paths_by_word:
        paths_by_word[word_idx] = enumerate_word_edge_paths_by_pattern(
            lp,
            word_idx,
            max_paths=max_paths,
            cache=path_pattern_cache,
        )


WORKER_STATE = {}


def init_worker(state):
    global WORKER_STATE
    WORKER_STATE = state


def pair_batch_worker(tasks):
    return [pair_worker(task) for task in tasks]


def triple_batch_worker(tasks):
    return [triple_worker(task) for task in tasks]


def pair_worker(task):
    state = WORKER_STATE
    lp = state["lp"]
    left_word, right_word, selected_tokens = task
    start = time.monotonic()
    result = pair_reduced_fractional_edge_hull_cut_from_paths(
        lp,
        state["f_values"],
        state["g_values"],
        state["t_values"],
        left_word,
        right_word,
        selected_tokens,
        state["paths_by_word"][left_word],
        state["paths_by_word"][right_word],
        tolerance=state["tolerance"],
        max_pair_rows=state["pair_row_limit"],
    )
    return normalize_cut_result("pair", (left_word, right_word), selected_tokens, result, time.monotonic() - start)


def triple_worker(task):
    state = WORKER_STATE
    left_word, middle_word, right_word, selected_tokens = task
    start = time.monotonic()
    result = triple_reduced_fractional_edge_hull_cut_from_paths(
        state["lp"],
        state["f_values"],
        state["g_values"],
        state["t_values"],
        (left_word, middle_word, right_word),
        selected_tokens,
        [state["paths_by_word"][word_idx] for word_idx in (left_word, middle_word, right_word)],
        tolerance=state["tolerance"],
        max_rows=state["triple_row_limit"],
    )
    return normalize_cut_result("triple", (left_word, middle_word, right_word), selected_tokens, result, time.monotonic() - start)


def normalize_cut_result(kind, words, selected_tokens, result, wall_seconds):
    row = {
        "kind": kind,
        "words": tuple(int(word) for word in words),
        "num_selected_tokens": len(selected_tokens),
        "wall_seconds": wall_seconds,
        "has_cut": False,
        "skip_reason": None,
    }
    if result is None:
        return row
    row.update(
        {
            "skip_reason": result.get("skip_reason"),
            "reduced_rows": result.get("reduced_rows"),
            "edge_vars": result.get("edge_vars"),
            "build_seconds": result.get("build_seconds"),
            "solve_seconds": result.get("solve_seconds"),
            "validation_seconds": result.get("validation_seconds"),
        }
    )
    if "violation" in result:
        row.update(
            {
                "has_cut": True,
                "violation": result["violation"],
                "rhs": result["rhs"],
                "num_edge_coefficients": len(result["edge_coefficients"]),
                "num_token_coefficients": len(result["token_coefficients"]),
            }
        )
    return row


def triple_reduced_fractional_edge_hull_cut_from_paths(lp, f_values, g_values, t_values, word_indices, selected_tokens, paths_list, *, tolerance, max_rows):
    num_f = lp["num_nonfree_edges"]
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    selected_set = set(selected_tokens)
    word_columns = []
    current_values = []
    for word_idx in word_indices:
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            value = float(f_values[edge_idx])
            if tolerance < value < 1.0 - tolerance:
                word_columns.append(int(edge_idx))
                current_values.append(value)
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            value = float(g_values[edge_idx])
            if tolerance < value < 1.0 - tolerance:
                word_columns.append(int(num_f + edge_idx))
                current_values.append(value)

    build_start = time.monotonic()
    col_position = {col_idx: idx for idx, col_idx in enumerate(word_columns)}
    signatures = [
        projected_pair_hull_path_signatures(paths, selected_set, selected_position, col_position)
        for paths in paths_list
    ]
    pair_keys = set()
    for first_edges, first_mask in signatures[0]:
        for second_edges, second_mask in signatures[1]:
            pair_keys.add((first_edges + second_edges, first_mask | second_mask))
    row_keys = []
    row_key_set = set()
    limit_hit = False
    for pair_edges, pair_mask in pair_keys:
        for third_edges, third_mask in signatures[2]:
            row_key = (pair_edges + third_edges, pair_mask | third_mask)
            if row_key in row_key_set:
                continue
            if max_rows > 0 and len(row_keys) >= max_rows:
                limit_hit = True
                break
            row_key_set.add(row_key)
            row_keys.append(row_key)
        if limit_hit:
            break

    num_edge_vars = len(word_columns)
    num_token_vars = len(selected_tokens)
    a_pos_offset = 0
    a_neg_offset = num_edge_vars
    b_pos_offset = 2 * num_edge_vars
    b_neg_offset = b_pos_offset + num_token_vars
    gamma_col = b_neg_offset + num_token_vars
    num_vars = gamma_col + 1

    if limit_hit:
        return {
            "skip_reason": "reduced_rows",
            "build_seconds": time.monotonic() - build_start,
            "solve_seconds": 0.0,
            "validation_seconds": 0.0,
            "reduced_rows": len(row_keys),
            "edge_vars": num_edge_vars,
        }

    rows = []
    cols = []
    data = []
    rhs = []
    for row_idx, (kept_positions, token_mask) in enumerate(row_keys):
        for pos in kept_positions:
            rows.extend((row_idx, row_idx))
            cols.extend((a_pos_offset + pos, a_neg_offset + pos))
            data.extend((1.0, -1.0))
        for pos in range(num_token_vars):
            rows.append(row_idx)
            cols.append(b_pos_offset + pos)
            data.append(1.0)
            if token_mask & (1 << pos):
                rows.append(row_idx)
                cols.append(b_neg_offset + pos)
                data.append(-1.0)
        rows.append(row_idx)
        cols.append(gamma_col)
        data.append(1.0)
        rhs.append(0.0)

    norm_row = len(row_keys)
    for pos in range(num_edge_vars):
        rows.extend((norm_row, norm_row))
        cols.extend((a_pos_offset + pos, a_neg_offset + pos))
        data.extend((1.0, 1.0))
    for pos in range(num_token_vars):
        rows.extend((norm_row, norm_row))
        cols.extend((b_pos_offset + pos, b_neg_offset + pos))
        data.extend((1.0, 1.0))
    rhs.append(1.0)

    constraints = sp.coo_matrix((data, (rows, cols)), shape=(norm_row + 1, num_vars), dtype=float).tocsr()
    objective = np.zeros(num_vars, dtype=float)
    for pos, value in enumerate(current_values):
        objective[a_pos_offset + pos] = -value
        objective[a_neg_offset + pos] = value
    for pos, token_idx in enumerate(selected_tokens):
        token_value = float(t_values[token_idx])
        objective[b_pos_offset + pos] = -token_value
        objective[b_neg_offset + pos] = token_value
    objective[gamma_col] = -1.0
    bounds = [(0.0, None)] * num_vars
    bounds[gamma_col] = (None, None)
    build_seconds = time.monotonic() - build_start
    solve_start = time.monotonic()
    result = linprog(c=objective, A_ub=constraints, b_ub=np.asarray(rhs, dtype=float), bounds=bounds, method="highs")
    solve_seconds = time.monotonic() - solve_start
    base = {
        "skip_reason": None,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "validation_seconds": 0.0,
        "reduced_rows": len(row_keys),
        "edge_vars": num_edge_vars,
    }
    if not result.success:
        base["skip_reason"] = "linprog"
        return base
    if -float(result.fun) <= tolerance:
        return base

    edge_coefficients = {}
    for col_idx, pos in col_position.items():
        coefficient = float(result.x[a_pos_offset + pos] - result.x[a_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            edge_coefficients[int(col_idx)] = coefficient
    token_coefficients = {}
    for token_idx, pos in selected_position.items():
        coefficient = float(result.x[b_pos_offset + pos] - result.x[b_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            token_coefficients[int(token_idx)] = coefficient
    rhs_value = -float(result.x[gamma_col])
    validation_start = time.monotonic()
    max_slack = projected_max_slack(edge_coefficients, token_coefficients, rhs_value, selected_tokens, row_keys, word_columns)
    current_violation = pair_hull_current_violation(
        edge_coefficients,
        token_coefficients,
        rhs_value,
        selected_tokens,
        f_values,
        g_values,
        t_values,
        num_f,
    )
    base["validation_seconds"] = time.monotonic() - validation_start
    if max_slack > 1e-7 or current_violation <= tolerance:
        base["skip_reason"] = "invalid"
        return base
    base.update(
        {
            "violation": float(current_violation),
            "edge_coefficients": edge_coefficients,
            "token_coefficients": token_coefficients,
            "rhs": rhs_value,
        }
    )
    return base


def projected_max_slack(edge_coefficients, token_coefficients, rhs, selected_tokens, row_keys, word_columns):
    positive_token_sum = sum(max(0.0, float(token_coefficients.get(token_idx, 0.0))) for token_idx in selected_tokens)
    max_slack = -float("inf")
    for kept_positions, token_mask in row_keys:
        lhs = positive_token_sum
        for pos in kept_positions:
            lhs += float(edge_coefficients.get(int(word_columns[pos]), 0.0))
        for pos, token_idx in enumerate(selected_tokens):
            if token_mask & (1 << pos):
                lhs += min(0.0, float(token_coefficients.get(token_idx, 0.0)))
        max_slack = max(max_slack, lhs - rhs)
    return max_slack


def run_tasks(tasks, worker_state, worker_fn, initializer, workers, batch_size, label, progress_interval):
    start = time.monotonic()
    if not tasks:
        return []
    worker_count = max(1, int(workers))
    records = []
    checked = 0
    if worker_count == 1:
        initializer(worker_state)
        for task in tasks:
            records.extend(worker_fn([task]))
            checked += 1
            if checked == 1 or checked % progress_interval == 0 or checked == len(tasks):
                log_progress(label, checked, len(tasks), records, start)
    else:
        context = mp.get_context("fork") if hasattr(os, "fork") else None
        batches = list(chunked(tasks, batch_size))
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context, initializer=initializer, initargs=(worker_state,)) as executor:
            futures = [executor.submit(worker_fn, batch) for batch in batches]
            for future in as_completed(futures):
                batch_records = future.result()
                records.extend(batch_records)
                checked += len(batch_records)
                if checked == len(batch_records) or checked % progress_interval < len(batch_records) or checked == len(tasks):
                    log_progress(label, checked, len(tasks), records, start)
    return records


def log_progress(label, checked, total, records, start):
    cuts = sum(1 for row in records if row.get("has_cut"))
    skipped = Counter(row.get("skip_reason") for row in records if row.get("skip_reason"))
    elapsed = time.monotonic() - start
    LOGGER.info(
        "%s frontier progress: checked=%d/%d cuts=%d skipped=%s elapsed=%.1fs rate=%.1f/s",
        label,
        checked,
        total,
        cuts,
        dict(skipped),
        elapsed,
        checked / elapsed if elapsed else 0.0,
    )


def write_cut_records(path, records, words):
    with path.open("w", encoding="utf-8") as handle:
        rank = 0
        for row in sorted((record for record in records if record.get("has_cut")), key=lambda item: item["violation"], reverse=True):
            rank += 1
            payload = dict(row)
            payload["rank"] = rank
            payload["word_strings"] = [words[word_idx] for word_idx in payload["words"]]
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def summarize_records(records, population_size):
    cuts = [row for row in records if row.get("has_cut")]
    skip_counts = Counter(row.get("skip_reason") for row in records if row.get("skip_reason"))
    cut_rate = len(cuts) / max(1, len(records))
    return {
        "checked": len(records),
        "cuts": len(cuts),
        "cut_rate": cut_rate,
        "population_size": population_size,
        "extrapolated_cuts": cut_rate * population_size if population_size is not None else None,
        "skipped": dict(skip_counts),
        "violation": quantiles([row["violation"] for row in cuts]),
        "reduced_rows": quantiles([row.get("reduced_rows", 0) for row in records]),
        "edge_vars": quantiles([row.get("edge_vars", 0) for row in records]),
        "wall_seconds": sum(float(row.get("wall_seconds", 0.0)) for row in records),
    }


def quantiles(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def format_summary(summary):
    pair = summary["pairs"]
    triple = summary["triples"]
    return [
        (
            "pair frontier: "
            f"checked={pair['checked']} cuts={pair['cuts']} rate={pair['cut_rate']:.5f} "
            f"population={pair['population_size']} extrapolated={pair['extrapolated_cuts']:.1f}"
            if pair["extrapolated_cuts"] is not None
            else f"pair frontier: checked={pair['checked']} cuts={pair['cuts']} rate={pair['cut_rate']:.5f}"
        ),
        f"pair violation={pair['violation']} skipped={pair['skipped']} rows={pair['reduced_rows']}",
        (
            "triple frontier: "
            f"checked={triple['checked']} cuts={triple['cuts']} rate={triple['cut_rate']:.5f} "
            f"population~={triple['population_size']} extrapolated~={triple['extrapolated_cuts']:.1f}"
            if triple["extrapolated_cuts"] is not None
            else f"triple frontier: checked={triple['checked']} cuts={triple['cuts']} rate={triple['cut_rate']:.5f}"
        ),
        f"triple violation={triple['violation']} skipped={triple['skipped']} rows={triple['reduced_rows']}",
    ]


if __name__ == "__main__":
    main()
