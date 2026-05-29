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
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
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
    pair_hull_max_projected_row_slack,
    prepare_lp_data,
    projected_pair_hull_path_signatures,
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
    start = time.monotonic()
    deadline = start + args.time_limit_seconds if args.time_limit_seconds > 0 else None

    LOGGER.info("Loading checkpoint from %s", state_dir)
    checkpoint = load_checkpoint_state(state_dir)
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
    tasks, prep_summary, paths_by_word = prepare_tasks(args, lp, f_values, t_values, checkpoint["existing_cut_keys"])
    prep_seconds = time.monotonic() - prep_start
    LOGGER.info("Prepared tasks=%d prep=%.1fs summary=%s", len(tasks), prep_seconds, prep_summary)

    records = run_scan(args, lp, words, tokens, f_values, g_values, t_values, tasks, paths_by_word, deadline)
    records_path = output_dir / "higher_order_scan_records.jsonl"
    write_jsonl(records_path, records)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_next_iteration": checkpoint.get("next_iteration"),
        "active_cuts": len(checkpoint["existing_cut_keys"]),
        "elapsed_seconds": time.monotonic() - start,
        "time_limit_seconds": args.time_limit_seconds,
        "prep": prep_summary | {"prep_seconds": prep_seconds},
        "scan": summarize_scan(records),
        "records_path": str(records_path),
    }
    summary_path = output_dir / "summary.json"
    log_path = output_dir / "summary.log"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = format_summary(summary)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        LOGGER.info(line)
    LOGGER.info("Wrote summary=%s records=%s", summary_path, records_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe 4- and 5-word reduced-edge hull LP cuts from a checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8)
    parser.add_argument("--cut-tolerance", type=float, default=1e-6)
    parser.add_argument("--arities", default="4,5")
    parser.add_argument("--max-words", type=int, default=700)
    parser.add_argument("--max-word-length", type=int, default=12)
    parser.add_argument("--top-words-per-color", type=int, default=48)
    parser.add_argument("--candidate-word-multiplier", type=float, default=16.0)
    parser.add_argument("--candidate-top-words-multiplier", type=float, default=16.0)
    parser.add_argument("--candidate-sample-per-arity", type=int, default=30000)
    parser.add_argument("--tasks-per-arity", type=int, default=12000)
    parser.add_argument("--row-limit", type=int, default=100000)
    parser.add_argument("--token-mode", choices=("shared_all", "at_least_two"), default="at_least_two")
    parser.add_argument("--min-fractional-colors", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=20260528)
    parser.add_argument("--time-limit-seconds", type=float, default=1800.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def load_checkpoint_state(state_dir: Path) -> dict:
    payload = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    payload["existing_cut_keys"] = {tupleify_json_key(key) for key in payload.get("existing_cut_keys", [])}
    return payload


def prepare_tasks(args, lp, f_values, t_values, existing_cut_keys):
    rng = random.Random(args.random_seed)
    candidate_max_words = max(args.max_words, int(math.ceil(args.max_words * max(1.0, args.candidate_word_multiplier))))
    candidate_top_words = max(
        args.top_words_per_color,
        int(math.ceil(args.top_words_per_color * max(1.0, args.candidate_top_words_multiplier))),
    )
    pair_rows, word_color_scores = short_word_pair_candidates(
        lp,
        f_values,
        t_values,
        max_words=candidate_max_words,
        max_word_length=args.max_word_length,
        top_words_per_color=candidate_top_words,
        tolerance=args.cut_tolerance,
    )
    pair_rows = list(pair_rows)
    graph = defaultdict(set)
    pair_scores = {}
    for score, left_word, right_word in pair_rows:
        graph[int(left_word)].add(int(right_word))
        graph[int(right_word)].add(int(left_word))
        pair_scores[tuple(sorted((int(left_word), int(right_word))))] = float(score)
    graph_words = sorted(graph)
    fractional_colors = set(np.flatnonzero((t_values > args.cut_tolerance) & (t_values < 1.0 - args.cut_tolerance)))
    arities = tuple(int(item) for item in args.arities.split(",") if item.strip())

    word_colors = {}
    path_pattern_cache = {}
    paths_by_word = {}
    tasks = []
    skipped = Counter()
    sampled_by_arity = Counter()
    existing_prefixes = {
        key[:3]
        for key in existing_cut_keys
        if len(key) >= 3 and key[0] in {"higher_order_hull_probe", "short_word_higher_order_hull"}
    }
    for arity in arities:
        sampled = sample_connected_word_sets(
            graph_words,
            graph,
            arity,
            sample_limit=args.candidate_sample_per_arity,
            rng=rng,
        )
        sampled_by_arity[arity] = len(sampled)
        arity_tasks = 0
        for word_indices in sampled:
            if arity_tasks >= args.tasks_per_arity:
                break
            if ("higher_order_hull_probe", arity, word_indices) in existing_prefixes:
                skipped[f"existing_{arity}"] += 1
                continue
            color_sets = [
                word_colors.setdefault(word_idx, set(all_word_token_colors(lp, word_idx)))
                for word_idx in word_indices
            ]
            if args.token_mode == "shared_all":
                selected_set = set.intersection(*color_sets) & fractional_colors
            else:
                counts = Counter(
                    token_idx
                    for colors_for_word in color_sets
                    for token_idx in colors_for_word
                    if token_idx in fractional_colors
                )
                selected_set = {token_idx for token_idx, count in counts.items() if count >= 2}
            selected_tokens = tuple(sorted(selected_set))
            if len(selected_tokens) < args.min_fractional_colors:
                skipped[f"shared_{arity}"] += 1
                continue
            ok = True
            for word_idx in word_indices:
                if word_idx not in paths_by_word:
                    paths_by_word[word_idx] = enumerate_word_edge_paths_by_pattern(
                        lp,
                        word_idx,
                        max_paths=args.max_paths,
                        cache=path_pattern_cache,
                    )
                if paths_by_word[word_idx] is None:
                    ok = False
                    break
            if not ok:
                skipped[f"paths_{arity}"] += 1
                continue
            tasks.append(
                {
                    "arity": arity,
                    "word_indices": word_indices,
                    "selected_tokens": selected_tokens,
                    "features": candidate_features(
                        lp,
                        f_values,
                        t_values,
                        word_indices,
                        selected_tokens,
                        word_color_scores,
                        word_colors,
                        paths_by_word,
                        pair_scores,
                        args.cut_tolerance,
                    ),
                }
            )
            arity_tasks += 1
    rng.shuffle(tasks)
    return tasks, {
        "candidate_words": candidate_max_words,
        "candidate_top_words": candidate_top_words,
        "candidate_pairs": len(pair_rows),
        "sampled_by_arity": dict(sampled_by_arity),
        "tasks_by_arity": dict(Counter(task["arity"] for task in tasks)),
        "skipped": dict(skipped),
        "path_patterns": len(path_pattern_cache),
    }, paths_by_word


def sample_connected_word_sets(graph_words, graph, arity, *, sample_limit, rng):
    if not graph_words or arity <= 1:
        return []
    samples = set()
    attempts = 0
    max_attempts = max(sample_limit * 50, 1000)
    while len(samples) < sample_limit and attempts < max_attempts:
        attempts += 1
        current = {rng.choice(graph_words)}
        frontier = set(graph[next(iter(current))])
        while len(current) < arity and frontier:
            next_word = rng.choice(tuple(frontier))
            current.add(next_word)
            frontier.update(graph[next_word])
            frontier.difference_update(current)
        if len(current) == arity:
            samples.add(tuple(sorted(current)))
    return list(samples)


def candidate_features(
    lp,
    f_values,
    t_values,
    word_indices,
    selected_tokens,
    word_color_scores,
    word_colors,
    paths_by_word,
    pair_scores,
    tolerance,
):
    color_sets = [word_colors[word_idx] for word_idx in word_indices]
    fractional_selected = set(selected_tokens)
    pair_shared = []
    pair_score_values = []
    for left_idx, right_idx in combinations(range(len(word_indices)), 2):
        shared = (color_sets[left_idx] & color_sets[right_idx]) & fractional_selected
        pair_shared.append(len(shared))
        pair_score_values.append(float(pair_scores.get(tuple(sorted((word_indices[left_idx], word_indices[right_idx]))), 0.0)))
    selected_edge_vars = 0
    fractional_edge_vars = 0
    fractional_edge_signal = 0.0
    for word_idx in word_indices:
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            edge_value = float(f_values[edge_idx])
            if not (tolerance < edge_value < 1.0 - tolerance):
                continue
            fractional_edge_vars += 1
            info = lp["nonfree_edge_info"][edge_idx]
            fractional_edge_signal += min(edge_value, 1.0 - edge_value) * max(1, int(info["end"]) - int(info["start"]))
            if int(info["token_index"]) in fractional_selected:
                selected_edge_vars += 1
    path_counts = [len(paths_by_word[word_idx]) for word_idx in word_indices]
    return {
        "selected_tokens": len(selected_tokens),
        "pair_shared_min": min(pair_shared) if pair_shared else 0,
        "pair_shared_max": max(pair_shared) if pair_shared else 0,
        "pair_shared_sum": sum(pair_shared),
        "positive_pair_shared": sum(1 for value in pair_shared if value > 0),
        "pair_score_sum": sum(pair_score_values),
        "pair_score_max": max(pair_score_values) if pair_score_values else 0.0,
        "all_shared": len(set.intersection(*color_sets) & fractional_selected),
        "word_weight_sum": sum(float(lp["word_weights"][word_idx]) for word_idx in word_indices),
        "word_length_sum": sum(int(lp["word_lengths"][word_idx]) for word_idx in word_indices),
        "word_length_max": max(int(lp["word_lengths"][word_idx]) for word_idx in word_indices),
        "nonfree_fractional_edge_vars": fractional_edge_vars,
        "selected_nonfree_fractional_edge_vars": selected_edge_vars,
        "fractional_edge_signal": fractional_edge_signal,
        "path_count_min": min(path_counts),
        "path_count_max": max(path_counts),
        "path_count_product_log10": sum(math.log10(max(1, count)) for count in path_counts),
    }


WORKER_STATE = {}


def init_worker(state):
    global WORKER_STATE
    WORKER_STATE = state


def scan_batch_worker(batch):
    return [scan_worker(task) for task in batch]


def scan_worker(task):
    state = WORKER_STATE
    lp = state["lp"]
    word_indices = task["word_indices"]
    selected_tokens = task["selected_tokens"]
    num_f = lp["num_nonfree_edges"]
    free_fractional_edges = 0
    for word_idx in word_indices:
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            edge_value = float(state["g_values"][edge_idx])
            if state["tolerance"] < edge_value < 1.0 - state["tolerance"]:
                free_fractional_edges += 1
    start = time.monotonic()
    result = ntuple_reduced_fractional_edge_hull_cut_from_paths(
        lp,
        state["f_values"],
        state["g_values"],
        state["t_values"],
        word_indices,
        selected_tokens,
        [state["paths_by_word"][word_idx] for word_idx in word_indices],
        tolerance=state["tolerance"],
        max_rows=state["row_limit"],
    )
    row = {
        "arity": int(task["arity"]),
        "words": word_indices,
        "word_strings": [state["words"][word_idx] for word_idx in word_indices],
        "selected_tokens": selected_tokens,
        "selected_token_strings": [state["tokens"][token_idx].token for token_idx in selected_tokens],
        "features": task["features"] | {"free_fractional_edge_vars": free_fractional_edges},
        "wall_seconds": time.monotonic() - start,
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
    if "edge_coefficients" not in result:
        return row
    edge_coefficients = result["edge_coefficients"]
    token_coefficients = result["token_coefficients"]
    row.update(
        {
            "has_cut": True,
            "violation": result["violation"],
            "rhs": result["rhs"],
            "reduced_objective_violation": result["reduced_violation"],
            "nonzero_edge_coefficients": len(edge_coefficients),
            "nonzero_token_coefficients": sum(1 for value in token_coefficients.values() if abs(float(value)) > 1e-10),
            "edge_support": edge_support_summary(lp, edge_coefficients),
            "coefficient_abs": coefficient_abs_summary(edge_coefficients, token_coefficients),
            "coefficient_pattern": coefficient_pattern(lp, word_indices, edge_coefficients, token_coefficients),
            "edge_coefficients": sorted((int(k), float(v)) for k, v in edge_coefficients.items()),
            "token_coefficients": sorted((int(k), float(v)) for k, v in token_coefficients.items()),
        }
    )
    return row


def ntuple_reduced_fractional_edge_hull_cut_from_paths(
    lp,
    f_values,
    g_values,
    t_values,
    word_indices,
    selected_tokens,
    paths_list,
    *,
    tolerance,
    max_rows,
):
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
    signatures.sort(key=len)
    row_keys = [(tuple(), 0)]
    limit_hit = False
    for signature in signatures:
        next_keys = []
        next_seen = set()
        for base_edges, base_mask in row_keys:
            for path_edges, path_mask in signature:
                row_key = (base_edges + path_edges, base_mask | path_mask)
                if row_key in next_seen:
                    continue
                if max_rows > 0 and len(next_keys) >= max_rows:
                    limit_hit = True
                    break
                next_seen.add(row_key)
                next_keys.append(row_key)
            if limit_hit:
                break
        row_keys = next_keys
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
    reduced_violation = -float(result.fun)
    if reduced_violation <= tolerance:
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
    max_slack = pair_hull_max_projected_row_slack(
        edge_coefficients,
        token_coefficients,
        rhs_value,
        selected_tokens,
        row_keys,
        word_columns,
    )
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
        base["skip_reason"] = "invalid_reduced_cut"
        return base
    base.update(
        {
            "violation": float(current_violation),
            "reduced_violation": float(reduced_violation),
            "edge_coefficients": edge_coefficients,
            "token_coefficients": token_coefficients,
            "rhs": float(rhs_value),
        }
    )
    return base


def run_scan(args, lp, words, tokens, f_values, g_values, t_values, tasks, paths_by_word, deadline):
    worker_state = {
        "lp": lp,
        "words": words,
        "tokens": tokens,
        "f_values": f_values,
        "g_values": g_values,
        "t_values": t_values,
        "paths_by_word": paths_by_word,
        "tolerance": args.cut_tolerance,
        "row_limit": args.row_limit,
    }
    worker_count = max(1, int(args.workers))
    batch_size = max(1, int(args.batch_size))
    records = []
    start = time.monotonic()
    if worker_count == 1:
        init_worker(worker_state)
        for task in tasks:
            if deadline is not None and time.monotonic() >= deadline:
                break
            records.append(scan_worker(task))
            maybe_log_scan_progress(records, len(tasks), start, args.progress_interval)
        return records

    context = mp.get_context("fork") if hasattr(os, "fork") else None
    batches = list(chunked(tasks, batch_size))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=init_worker,
        initargs=(worker_state,),
    ) as executor:
        in_flight = set()
        batch_iter = iter(batches)
        max_in_flight = max(1, worker_count * 3)

        def submit_more() -> None:
            while len(in_flight) < max_in_flight:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    return
                in_flight.add(executor.submit(scan_batch_worker, batch))

        submit_more()
        while in_flight:
            try:
                for future in as_completed(in_flight, timeout=5.0):
                    in_flight.remove(future)
                    records.extend(future.result())
                    maybe_log_scan_progress(records, len(tasks), start, args.progress_interval)
                    submit_more()
                    break
            except TimeoutError:
                pass
            if deadline is not None and time.monotonic() >= deadline:
                for future in in_flight:
                    future.cancel()
                break
    return records


def maybe_log_scan_progress(records, total, start, progress_interval):
    checked = len(records)
    if checked == total or (progress_interval > 0 and checked % progress_interval < 32):
        cuts = sum(1 for row in records if row.get("has_cut"))
        by_arity = Counter(row["arity"] for row in records)
        skipped = Counter(row.get("skip_reason") for row in records if row.get("skip_reason"))
        LOGGER.info(
            "scan progress: checked=%d/%d by_arity=%s cuts=%d skipped=%s elapsed=%.1fs",
            checked,
            total,
            dict(by_arity),
            cuts,
            dict(skipped),
            time.monotonic() - start,
        )


def edge_support_summary(lp, edge_coefficients):
    num_f = lp["num_nonfree_edges"]
    by_word = Counter()
    by_kind = Counter()
    by_sign = Counter()
    token_indices = set()
    for col_idx, coefficient in edge_coefficients.items():
        if col_idx < num_f:
            info = lp["nonfree_edge_info"][col_idx]
            by_kind["nonfree"] += 1
            token_indices.add(int(info["token_index"]))
        else:
            info = lp["free_edge_info"][col_idx - num_f]
            by_kind["free"] += 1
        by_word[int(info["word_idx"])] += 1
        by_sign["positive" if coefficient > 0 else "negative"] += 1
    return {
        "by_word": dict(by_word),
        "by_kind": dict(by_kind),
        "by_sign": dict(by_sign),
        "distinct_nonfree_edge_tokens": len(token_indices),
    }


def coefficient_abs_summary(edge_coefficients, token_coefficients):
    return {
        "edge": quantiles([abs(float(value)) for value in edge_coefficients.values()]),
        "token": quantiles([abs(float(value)) for value in token_coefficients.values()]),
    }


def coefficient_pattern(lp, word_indices, edge_coefficients, token_coefficients):
    num_f = lp["num_nonfree_edges"]
    by_word = Counter()
    positive_nonfree_by_word = Counter()
    negative_nonfree_by_word = Counter()
    positive_free_by_word = Counter()
    negative_free_by_word = Counter()
    used_nonfree_tokens = set()
    for col_idx, coefficient in edge_coefficients.items():
        if col_idx < num_f:
            info = lp["nonfree_edge_info"][col_idx]
            used_nonfree_tokens.add(int(info["token_index"]))
            if coefficient > 0:
                positive_nonfree_by_word[int(info["word_idx"])] += 1
            else:
                negative_nonfree_by_word[int(info["word_idx"])] += 1
        else:
            info = lp["free_edge_info"][col_idx - num_f]
            if coefficient > 0:
                positive_free_by_word[int(info["word_idx"])] += 1
            else:
                negative_free_by_word[int(info["word_idx"])] += 1
        by_word[int(info["word_idx"])] += 1
    token_signs = Counter("positive" if value > 0 else "negative" for value in token_coefficients.values())
    edge_abs_values = sorted({round(abs(float(value)), 6) for value in edge_coefficients.values() if abs(float(value)) > 1e-10})
    token_abs_values = sorted({round(abs(float(value)), 6) for value in token_coefficients.values() if abs(float(value)) > 1e-10})
    return {
        "edge_count_by_tuple_word": tuple(int(by_word.get(word_idx, 0)) for word_idx in word_indices),
        "positive_nonfree_by_tuple_word": tuple(int(positive_nonfree_by_word.get(word_idx, 0)) for word_idx in word_indices),
        "negative_nonfree_by_tuple_word": tuple(int(negative_nonfree_by_word.get(word_idx, 0)) for word_idx in word_indices),
        "positive_free_by_tuple_word": tuple(int(positive_free_by_word.get(word_idx, 0)) for word_idx in word_indices),
        "negative_free_by_tuple_word": tuple(int(negative_free_by_word.get(word_idx, 0)) for word_idx in word_indices),
        "token_signs": dict(token_signs),
        "distinct_used_nonfree_tokens": len(used_nonfree_tokens),
        "edge_abs_values": edge_abs_values,
        "token_abs_values": token_abs_values,
    }


def summarize_scan(records):
    by_arity = {}
    for arity in sorted({row["arity"] for row in records}):
        rows = [row for row in records if row["arity"] == arity]
        hits = [row for row in rows if row.get("has_cut")]
        solved = [row for row in rows if row.get("skip_reason") is None]
        by_arity[str(arity)] = summarize_rows(rows, hits, solved)
    hits = [row for row in records if row.get("has_cut")]
    return {
        "checked": len(records),
        "cuts": len(hits),
        "cut_rate": len(hits) / max(1, len(records)),
        "by_arity": by_arity,
        "top_hits": top_hit_summary(hits, 20),
        "pattern_counts": pattern_counts(hits),
    }


def summarize_rows(rows, hits, solved):
    skipped = Counter(row.get("skip_reason") for row in rows if row.get("skip_reason"))
    return {
        "checked": len(rows),
        "solved_not_row_capped": len(solved),
        "cuts": len(hits),
        "cut_rate": len(hits) / max(1, len(rows)),
        "solved_cut_rate": len(hits) / max(1, len(solved)),
        "skipped": dict(skipped),
        "violation": quantiles([row["violation"] for row in hits]),
        "reduced_rows": quantiles([row.get("reduced_rows", 0) for row in rows if row.get("reduced_rows") is not None]),
        "edge_vars": quantiles([row.get("edge_vars", 0) for row in rows if row.get("edge_vars") is not None]),
        "hit_nonzero_edge_coefficients": quantiles([row["nonzero_edge_coefficients"] for row in hits]),
        "hit_nonzero_token_coefficients": quantiles([row["nonzero_token_coefficients"] for row in hits]),
    }


def top_hit_summary(hits, limit):
    return [
        {
            "arity": row["arity"],
            "word_strings": row["word_strings"],
            "selected_token_strings": row["selected_token_strings"],
            "violation": row["violation"],
            "rhs": row["rhs"],
            "reduced_rows": row.get("reduced_rows"),
            "edge_vars": row.get("edge_vars"),
            "nonzero_edge_coefficients": row["nonzero_edge_coefficients"],
            "nonzero_token_coefficients": row["nonzero_token_coefficients"],
            "coefficient_pattern": row["coefficient_pattern"],
            "features": row["features"],
        }
        for row in sorted(hits, key=lambda item: item["violation"], reverse=True)[:limit]
    ]


def pattern_counts(hits):
    counters = {
        "edge_count_by_tuple_word": Counter(),
        "positive_nonfree_by_tuple_word": Counter(),
        "token_signs": Counter(),
        "edge_abs_values": Counter(),
        "token_abs_values": Counter(),
    }
    for row in hits:
        pattern = row["coefficient_pattern"]
        for name in counters:
            counters[name][repr(pattern[name])] += 1
    return {name: dict(counter.most_common(20)) for name, counter in counters.items()}


def format_summary(summary):
    scan = summary["scan"]
    lines = [
        f"higher-order scan: checked={scan['checked']} cuts={scan['cuts']} cut_rate={scan['cut_rate']:.5f} "
        f"elapsed={summary['elapsed_seconds']:.1f}s",
        f"prep: {summary['prep']}",
    ]
    for arity, row in scan["by_arity"].items():
        lines.append(
            f"arity {arity}: checked={row['checked']} solved={row['solved_not_row_capped']} cuts={row['cuts']} "
            f"cut_rate={row['cut_rate']:.5f} solved_cut_rate={row['solved_cut_rate']:.5f} skipped={row['skipped']}"
        )
        lines.append(f"arity {arity}: violation={row['violation']} rows={row['reduced_rows']} edge_vars={row['edge_vars']}")
        lines.append(
            f"arity {arity}: hit_edges={row['hit_nonzero_edge_coefficients']} "
            f"hit_tokens={row['hit_nonzero_token_coefficients']}"
        )
    lines.append(f"patterns: {scan['pattern_counts']}")
    for hit in scan["top_hits"][:10]:
        lines.append(
            f"top hit arity={hit['arity']} violation={hit['violation']:.6g} rhs={hit['rhs']:.6g} "
            f"words={hit['word_strings']} tokens={hit['selected_token_strings']} pattern={hit['coefficient_pattern']}"
        )
    return lines


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def quantiles(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "max": float(np.max(arr)),
    }


if __name__ == "__main__":
    main()
