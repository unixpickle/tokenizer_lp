from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    all_word_token_colors,
    build_standard_form,
    count_pretokenized_strings,
    enumerate_word_edge_paths_by_pattern,
    pair_full_upward_hull_cut_from_paths,
    pair_hull_solution_cache_from_json,
    prepare_lp_data,
    reorder_short_word_pair_candidates,
    separate_short_word_pair_hull_cut_specs,
    separate_short_word_full_hull_cut_specs,
    short_word_pair_candidates,
    tupleify_json_key,
)
from tokenisation_lp.pretokenization import (
    DEFAULT_SPECIAL_TOKENS,
    build_pretokenizer,
    byte_level_alphabet,
)


LOGGER = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    run_dir = Path(args.run_dir).expanduser()
    state_dir = run_dir / "lp" / "training_state"
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading checkpoint from %s", state_dir)
    checkpoint = load_checkpoint_state(state_dir)
    latest_solution = np.load(state_dir / "latest_solution.npy", allow_pickle=False)
    existing_cut_keys = checkpoint["existing_cut_keys"]

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
    assert_solution_shape(lp, latest_solution)

    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    f_values = latest_solution[:num_f]
    g_values = latest_solution[num_f : num_f + num_g]
    t_values = latest_solution[num_f + num_g :]

    summary = {
        "run_dir": str(run_dir),
        "state_dir": str(state_dir),
        "checkpoint_next_iteration": checkpoint.get("next_iteration"),
        "checkpoint_completed": checkpoint.get("completed"),
        "active_cuts": len(existing_cut_keys),
        "lp": {
            "num_tokens": int(lp["num_tokens"]),
            "num_nonfree_edges": int(lp["num_nonfree_edges"]),
            "num_free_edges": int(lp["num_free_edges"]),
            "latest_solution_objective_from_edges": objective_from_solution(lp, latest_solution),
        },
    }

    full_hull_cuts = separate_short_word_full_hull_cut_specs(
        lp,
        f_values,
        g_values,
        t_values,
        existing_cut_keys=existing_cut_keys,
        tolerance=args.cut_tolerance,
        max_words=args.short_word_full_hull_max_words,
        max_word_length=args.short_word_full_hull_max_length,
        max_colors=args.short_word_full_hull_max_colors,
        max_paths=args.max_paths,
    )
    summary["next_short_word_full_hull_cuts"] = len(full_hull_cuts)
    LOGGER.info("Next short_word_full_hull cuts=%d", len(full_hull_cuts))

    cuts_path = output_dir / "next_pair_cuts.jsonl"
    reanalyzed_cuts_path = output_dir / "reanalyzed_pair_cuts.jsonl"
    analysis_path = output_dir / "pair_structure_analysis.json"
    text_log_path = output_dir / "analysis.log"
    log_lines = []

    next_pair_cuts = separate_short_word_pair_hull_cut_specs(
        lp,
        f_values,
        g_values,
        t_values,
        existing_cut_keys=existing_cut_keys,
        tolerance=args.cut_tolerance,
        max_words=args.pair_max_words,
        max_word_length=args.pair_max_word_length,
        max_colors=args.pair_max_colors,
        max_pair_rows=args.pair_max_pair_rows,
        max_pairs=args.pair_max_pairs,
        top_words_per_color=args.pair_top_words_per_color,
        candidate_word_multiplier=args.pair_candidate_word_multiplier,
        candidate_top_words_multiplier=args.pair_candidate_top_words_multiplier,
        candidate_strategy=args.pair_candidate_strategy,
        candidate_random_seed=args.pair_candidate_random_seed + 1_000_003 * args.separation_round,
        max_paths=args.max_paths,
        workers=args.pair_workers,
        batch_size=args.pair_batch_size,
        min_fractional_shared_colors=args.pair_min_fractional_shared_colors,
        solution_cache=checkpoint["short_word_pair_hull_solution_cache"],
        solution_cache_max_entries=args.pair_cache_max_entries,
        solution_cache_value_quantum=args.pair_cache_value_quantum,
    )
    next_pair_cuts.sort(key=lambda cut: cut[0], reverse=True)
    cut_pairs = {(cut[1][1], cut[1][2]) for cut in next_pair_cuts if len(cut[1]) >= 3}
    with reanalyzed_cuts_path.open("w", encoding="utf-8") as cuts_file:
        for cut_rank, cut in enumerate(next_pair_cuts, start=1):
            cuts_file.write(json.dumps(cut_record(cut_rank, cut, words), ensure_ascii=False) + "\n")
    summary["next_pair_cuts"] = len(next_pair_cuts)
    summary["next_pair_cut_log"] = str(cuts_path)
    LOGGER.info("Next short_word_pair_hull cuts=%d log=%s", len(next_pair_cuts), cuts_path)

    pair_rows, word_color_scores = build_pair_rows(args, lp, f_values, t_values)
    if cut_pairs:
        pair_rows = sorted(pair_rows, key=lambda row: ((row[1], row[2]) not in cut_pairs, -row[0]))
    tasks, skipped, paths_by_word, pattern_count = prepare_tasks(
        args,
        lp,
        pair_rows,
        t_values,
        existing_cut_keys,
    )
    summary["candidate_pairs"] = len(pair_rows)
    summary["prepared_tasks"] = len(tasks)
    summary["prepare_skipped"] = dict(skipped)
    summary["path_patterns"] = pattern_count
    LOGGER.info(
        "Prepared next pair round: candidates=%d tasks=%d skipped=%s patterns=%d",
        len(pair_rows),
        len(tasks),
        dict(skipped),
        pattern_count,
    )

    start = time.monotonic()
    records = []
    with cuts_path.open("w", encoding="utf-8") as cuts_file:
        for idx, task in enumerate(tasks, start=1):
            record = analyze_task(
                args,
                lp,
                words,
                freqs,
                tokens,
                f_values,
                g_values,
                t_values,
                word_color_scores,
                task,
                paths_by_word,
                include_variants=idx <= args.variant_tasks,
            )
            records.append(record)
            if record["has_cut"]:
                cuts_file.write(json.dumps(record["cut"], ensure_ascii=False) + "\n")
                cuts_file.flush()
            if idx == 1 or idx % args.progress_interval == 0:
                LOGGER.info(
                    "checked=%d/%d cuts=%d elapsed=%.1fs",
                    idx,
                    len(tasks),
                    sum(1 for row in records if row["has_cut"]),
                    time.monotonic() - start,
                )

    summary["checked_tasks"] = len(records)
    summary["reanalyzed_pair_cuts"] = sum(1 for row in records if row["has_cut"])
    summary["reanalyzed_pair_cut_log"] = str(reanalyzed_cuts_path)
    summary["elapsed_seconds"] = time.monotonic() - start
    report = {
        "summary": summary,
        "feature_summary": feature_summary(records),
        "heuristics": heuristic_tables(records),
        "dimension_reduction": dimension_reduction_summary(records),
        "examples": examples(records, args.examples_per_class),
    }
    analysis_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    log_lines.extend(format_log(report))
    text_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    for line in log_lines:
        LOGGER.info(line)
    LOGGER.info("Wrote cuts=%s analysis=%s log=%s", cuts_path, analysis_path, text_log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the next short_word_pair_hull separation round from a saved LP checkpoint."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8)
    parser.add_argument("--cut-tolerance", type=float, default=1e-6)
    parser.add_argument("--separation-round", type=int, default=10)
    parser.add_argument("--cuts-per-round", type=int, default=1000)
    parser.add_argument("--short-word-full-hull-max-words", type=int, default=12000)
    parser.add_argument("--short-word-full-hull-max-length", type=int, default=12)
    parser.add_argument("--short-word-full-hull-max-colors", type=int, default=96)
    parser.add_argument("--pair-max-words", type=int, default=700)
    parser.add_argument("--pair-max-word-length", type=int, default=12)
    parser.add_argument("--pair-max-colors", type=int, default=96)
    parser.add_argument("--pair-max-pair-rows", type=int, default=250000)
    parser.add_argument("--pair-max-pairs", type=int, default=5000)
    parser.add_argument("--pair-top-words-per-color", type=int, default=48)
    parser.add_argument("--pair-candidate-word-multiplier", type=float, default=4.0)
    parser.add_argument("--pair-candidate-top-words-multiplier", type=float, default=4.0)
    parser.add_argument("--pair-candidate-strategy", choices=("score", "mixed"), default="mixed")
    parser.add_argument("--pair-candidate-random-seed", type=int, default=0)
    parser.add_argument("--pair-min-fractional-shared-colors", type=int, default=2)
    parser.add_argument("--pair-workers", type=int, default=8)
    parser.add_argument("--pair-batch-size", type=int, default=128)
    parser.add_argument("--pair-cache-max-entries", type=int, default=500000)
    parser.add_argument("--pair-cache-value-quantum", type=float, default=1e-4)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--analysis-tasks", type=int, default=800)
    parser.add_argument("--variant-tasks", type=int, default=40)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--examples-per-class", type=int, default=12)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def load_checkpoint_state(state_dir: Path) -> dict:
    payload = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    payload["existing_cut_keys"] = {
        tupleify_json_key(key)
        for key in payload.get("existing_cut_keys", [])
    }
    payload["short_word_pair_hull_solution_cache"] = pair_hull_solution_cache_from_json(
        payload.get("short_word_pair_hull_solution_cache") or {}
    )
    return payload


def assert_solution_shape(lp, solution: np.ndarray) -> None:
    expected = lp["num_nonfree_edges"] + lp["num_free_edges"] + lp["num_tokens"]
    if solution.shape != (expected,):
        raise ValueError(f"latest_solution shape {solution.shape} does not match LP vars {expected}")


def objective_from_solution(lp, solution: np.ndarray) -> float:
    return float(np.dot(lp["c"], solution))


def build_pair_rows(args, lp, f_values, t_values):
    candidate_max_words = max(
        args.pair_max_words,
        int(math.ceil(args.pair_max_words * max(1.0, args.pair_candidate_word_multiplier))),
    )
    candidate_top_words_per_color = max(
        args.pair_top_words_per_color,
        int(math.ceil(args.pair_top_words_per_color * max(1.0, args.pair_candidate_top_words_multiplier))),
    )
    pair_rows, word_color_scores = short_word_pair_candidates(
        lp,
        f_values,
        t_values,
        max_words=candidate_max_words,
        max_word_length=args.pair_max_word_length,
        top_words_per_color=candidate_top_words_per_color,
        tolerance=args.cut_tolerance,
    )
    pair_rows = reorder_short_word_pair_candidates(
        pair_rows,
        lp,
        t_values,
        max_pairs=args.pair_max_pairs,
        tolerance=args.cut_tolerance,
        strategy=args.pair_candidate_strategy,
        random_seed=args.pair_candidate_random_seed + 1_000_003 * args.separation_round,
    )
    return pair_rows, word_color_scores


def cut_record(cut_rank, cut, words):
    violation, key, entries, rhs = cut
    left_word = key[1]
    right_word = key[2]
    return {
        "rank": int(cut_rank),
        "violation": float(violation),
        "key": jsonify_key(key),
        "left_word_idx": int(left_word),
        "right_word_idx": int(right_word),
        "left_word": words[left_word],
        "right_word": words[right_word],
        "rhs": float(rhs),
        "num_entries": len(entries),
    }


def prepare_tasks(args, lp, pair_rows, t_values, existing_cut_keys):
    path_pattern_cache = {}
    paths_by_word = {}
    colors_by_word = {}
    tasks = []
    skipped = Counter()
    fractional_colors = set(np.flatnonzero((t_values > args.cut_tolerance) & (t_values < 1.0 - args.cut_tolerance)))
    for original_rank, (score, left_word, right_word) in enumerate(pair_rows, start=1):
        if len(tasks) >= args.analysis_tasks:
            break
        full_key_prefix = ("short_word_pair_hull", left_word, right_word)
        if any(key[:3] == full_key_prefix for key in existing_cut_keys):
            skipped["existing"] += 1
            continue
        left_colors = colors_by_word.setdefault(left_word, set(all_word_token_colors(lp, left_word)))
        right_colors = colors_by_word.setdefault(right_word, set(all_word_token_colors(lp, right_word)))
        shared_fractional = sorted((left_colors & right_colors) & fractional_colors)
        if len(shared_fractional) < args.pair_min_fractional_shared_colors:
            skipped["shared"] += 1
            continue
        selected_tokens = tuple(sorted(left_colors | right_colors))
        if len(selected_tokens) < 2 or len(selected_tokens) > args.pair_max_colors:
            skipped["colors"] += 1
            continue
        left_paths = paths_by_word.setdefault(
            left_word,
            enumerate_word_edge_paths_by_pattern(
                lp,
                left_word,
                max_paths=args.max_paths,
                cache=path_pattern_cache,
            ),
        )
        right_paths = paths_by_word.setdefault(
            right_word,
            enumerate_word_edge_paths_by_pattern(
                lp,
                right_word,
                max_paths=args.max_paths,
                cache=path_pattern_cache,
            ),
        )
        if left_paths is None or right_paths is None:
            skipped["paths"] += 1
            continue
        pair_rows_count = len(left_paths) * len(right_paths)
        if pair_rows_count > args.pair_max_pair_rows:
            skipped["rows"] += 1
            continue
        tasks.append(
            {
                "rank": len(tasks) + 1,
                "original_rank": original_rank,
                "score": float(score),
                "left_word": int(left_word),
                "right_word": int(right_word),
                "selected_tokens": selected_tokens,
                "shared_fractional_tokens": tuple(shared_fractional),
                "left_paths": len(left_paths),
                "right_paths": len(right_paths),
                "pair_rows": int(pair_rows_count),
            }
        )
    return tasks, skipped, paths_by_word, len(path_pattern_cache)


def analyze_task(
    args,
    lp,
    words,
    freqs,
    tokens,
    f_values,
    g_values,
    t_values,
    word_color_scores,
    task,
    paths_by_word,
    *,
    include_variants: bool,
):
    left_word = task["left_word"]
    right_word = task["right_word"]
    selected_tokens = task["selected_tokens"]
    left_paths = paths_by_word[left_word]
    right_paths = paths_by_word[right_word]
    base = enrich_task(task, lp, words, freqs, tokens, f_values, t_values, word_color_scores)

    start = time.monotonic()
    cut = pair_full_upward_hull_cut_from_paths(
        lp,
        f_values,
        g_values,
        t_values,
        left_word,
        right_word,
        selected_tokens,
        left_paths,
        right_paths,
        tolerance=args.cut_tolerance,
    )
    base["full_solve_wall_seconds"] = time.monotonic() - start
    if cut is None:
        base.update({"has_cut": False, "violation": 0.0})
    else:
        violation, edge_coefficients, token_coefficients, rhs, coefficient_key, build_seconds, solve_seconds = cut
        base.update(
            {
                "has_cut": True,
                "violation": float(violation),
                "build_seconds": float(build_seconds),
                "solve_seconds": float(solve_seconds),
                "num_edge_coefficients": len(edge_coefficients),
                "num_token_coefficients": len(token_coefficients),
                "cut": {
                    "rank": task["rank"],
                    "original_rank": task["original_rank"],
                    "left_word": words[left_word],
                    "right_word": words[right_word],
                    "left_word_idx": left_word,
                    "right_word_idx": right_word,
                    "violation": float(violation),
                    "rhs": float(rhs),
                    "coefficient_key": jsonify_key(coefficient_key),
                    "num_edge_coefficients": len(edge_coefficients),
                    "num_token_coefficients": len(token_coefficients),
                    "pair_rows": task["pair_rows"],
                },
            }
        )

    if include_variants:
        base["dimension_variants"] = analyze_dimension_variants(
            args,
            lp,
            f_values,
            g_values,
            t_values,
            left_word,
            right_word,
            selected_tokens,
            left_paths,
            right_paths,
            task,
        )
    else:
        base["dimension_variants"] = []
    return base


def enrich_task(task, lp, words, freqs, tokens, f_values, t_values, word_color_scores):
    left_word = task["left_word"]
    right_word = task["right_word"]
    selected_tokens = task["selected_tokens"]
    shared_fractional = task["shared_fractional_tokens"]
    left_colors = set(all_word_token_colors(lp, left_word))
    right_colors = set(all_word_token_colors(lp, right_word))
    fractional_union = [
        token_idx
        for token_idx in selected_tokens
        if 1e-6 < float(t_values[token_idx]) < 1.0 - 1e-6
    ]
    return {
        **{key: serializable(value) for key, value in task.items() if key != "selected_tokens"},
        "selected_tokens": [int(token_idx) for token_idx in selected_tokens],
        "left_word_text": words[left_word],
        "right_word_text": words[right_word],
        "left_frequency": int(freqs[left_word]),
        "right_frequency": int(freqs[right_word]),
        "left_length": int(lp["word_lengths"][left_word]),
        "right_length": int(lp["word_lengths"][right_word]),
        "num_selected_colors": len(selected_tokens),
        "num_shared_colors": len(left_colors & right_colors),
        "num_fractional_union_colors": len(fractional_union),
        "num_fractional_shared_colors": len(shared_fractional),
        "shared_fractional_mass": float(sum(t_values[token_idx] for token_idx in shared_fractional)),
        "union_fractional_mass": float(sum(t_values[token_idx] for token_idx in fractional_union)),
        "shared_candidate_score": float(
            sum(
                min(
                    word_color_scores[left_word].get(token_idx, 0.0),
                    word_color_scores[right_word].get(token_idx, 0.0),
                )
                for token_idx in shared_fractional
            )
        ),
    }


def analyze_dimension_variants(args, lp, f_values, g_values, t_values, left_word, right_word, selected_tokens, left_paths, right_paths, task):
    variants = []
    fractional_union = tuple(
        token_idx
        for token_idx in selected_tokens
        if args.cut_tolerance < float(t_values[token_idx]) < 1.0 - args.cut_tolerance
    )
    shared_fractional = tuple(task["shared_fractional_tokens"])
    for name, token_subset in [
        ("all_tokens", selected_tokens),
        ("fractional_union_tokens", fractional_union),
        ("shared_fractional_tokens", shared_fractional),
    ]:
        if len(token_subset) < 2:
            continue
        variants.append(
            solve_variant(
                lp,
                f_values,
                g_values,
                t_values,
                left_word,
                right_word,
                token_subset,
                left_paths,
                right_paths,
                row_limit=None,
                tolerance=args.cut_tolerance,
                name=name,
            )
        )

    for row_limit in (128, 512, 2048):
        if task["pair_rows"] <= row_limit:
            continue
        variants.append(
            solve_variant(
                lp,
                f_values,
                g_values,
                t_values,
                left_word,
                right_word,
                selected_tokens,
                left_paths,
                right_paths,
                row_limit=row_limit,
                tolerance=args.cut_tolerance,
                name=f"all_tokens_first_{row_limit}_rows",
            )
        )
    return variants


def solve_variant(
    lp,
    f_values,
    g_values,
    t_values,
    left_word,
    right_word,
    selected_tokens,
    left_paths,
    right_paths,
    *,
    row_limit,
    tolerance,
    name,
):
    start = time.monotonic()
    result = solve_pair_cut_variant(
        lp,
        f_values,
        g_values,
        t_values,
        left_word,
        right_word,
        selected_tokens,
        left_paths,
        right_paths,
        row_limit=row_limit,
    )
    wall = time.monotonic() - start
    if result is None:
        return {
            "name": name,
            "tokens": len(selected_tokens),
            "row_limit": row_limit,
            "rows_built": min(len(left_paths) * len(right_paths), row_limit or len(left_paths) * len(right_paths)),
            "has_cut": False,
            "full_valid": False,
            "wall_seconds": wall,
        }
    violation, edge_coefficients, token_coefficients, rhs, rows_built = result
    full_slack_max = max_full_row_slack(
        lp,
        edge_coefficients,
        token_coefficients,
        rhs,
        selected_tokens,
        left_paths,
        right_paths,
    )
    current_violation = current_cut_violation(
        edge_coefficients,
        token_coefficients,
        rhs,
        selected_tokens,
        f_values,
        g_values,
        t_values,
        lp["num_nonfree_edges"],
    )
    return {
        "name": name,
        "tokens": len(selected_tokens),
        "row_limit": row_limit,
        "rows_built": rows_built,
        "has_cut": bool(violation > tolerance),
        "reduced_violation": float(violation),
        "current_violation": float(current_violation),
        "full_valid": bool(full_slack_max <= 1e-7),
        "max_full_row_slack": float(full_slack_max),
        "edge_coefficients": len(edge_coefficients),
        "token_coefficients": len(token_coefficients),
        "wall_seconds": wall,
    }


def solve_pair_cut_variant(lp, f_values, g_values, t_values, left_word_idx, right_word_idx, selected_tokens, left_paths, right_paths, *, row_limit):
    num_f = lp["num_nonfree_edges"]
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    selected_set = set(selected_tokens)
    word_columns = []
    current_values = []
    for word_idx in (left_word_idx, right_word_idx):
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            word_columns.append(edge_idx)
            current_values.append(float(f_values[edge_idx]))
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            word_columns.append(num_f + edge_idx)
            current_values.append(float(g_values[edge_idx]))

    col_position = {col_idx: idx for idx, col_idx in enumerate(word_columns)}
    num_edge_vars = len(word_columns)
    num_token_vars = len(selected_tokens)
    a_pos_offset = 0
    a_neg_offset = num_edge_vars
    b_pos_offset = 2 * num_edge_vars
    b_neg_offset = b_pos_offset + num_token_vars
    gamma_col = b_neg_offset + num_token_vars
    num_vars = gamma_col + 1

    rows = []
    cols = []
    data = []
    rhs = []
    row_idx = 0
    stop = False
    for left_columns, left_tokens in left_paths:
        left_required = set(left_tokens) & selected_set
        for right_columns, right_tokens in right_paths:
            if row_limit is not None and row_idx >= row_limit:
                stop = True
                break
            required = left_required | (set(right_tokens) & selected_set)
            for col_idx in (*left_columns, *right_columns):
                pos = col_position[col_idx]
                rows.extend((row_idx, row_idx))
                cols.extend((a_pos_offset + pos, a_neg_offset + pos))
                data.extend((1.0, -1.0))
            for pos in range(num_token_vars):
                rows.append(row_idx)
                cols.append(b_pos_offset + pos)
                data.append(1.0)
            for token_idx in required:
                rows.append(row_idx)
                cols.append(b_neg_offset + selected_position[token_idx])
                data.append(-1.0)
            rows.append(row_idx)
            cols.append(gamma_col)
            data.append(1.0)
            rhs.append(0.0)
            row_idx += 1
        if stop:
            break

    norm_row = row_idx
    for pos in range(num_edge_vars):
        rows.extend((norm_row, norm_row))
        cols.extend((a_pos_offset + pos, a_neg_offset + pos))
        data.extend((1.0, 1.0))
    for pos in range(num_token_vars):
        rows.extend((norm_row, norm_row))
        cols.extend((b_pos_offset + pos, b_neg_offset + pos))
        data.extend((1.0, 1.0))
    rhs.append(1.0)

    constraints = sp.coo_matrix((data, (rows, cols)), shape=(row_idx + 1, num_vars), dtype=float).tocsr()
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
    result = linprog(c=objective, A_ub=constraints, b_ub=np.array(rhs, dtype=float), bounds=bounds, method="highs")
    if not result.success:
        return None

    edge_coefficients = {}
    for col_idx, pos in col_position.items():
        coefficient = float(result.x[a_pos_offset + pos] - result.x[a_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            edge_coefficients[col_idx] = coefficient
    token_coefficients = {}
    for token_idx, pos in selected_position.items():
        coefficient = float(result.x[b_pos_offset + pos] - result.x[b_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            token_coefficients[token_idx] = coefficient
    gamma = float(result.x[gamma_col])
    return -float(result.fun), edge_coefficients, token_coefficients, -gamma, row_idx


def max_full_row_slack(lp, edge_coefficients, token_coefficients, rhs, selected_tokens, left_paths, right_paths):
    selected_set = set(selected_tokens)
    positive_token_sum = sum(max(0.0, float(token_coefficients.get(token_idx, 0.0))) for token_idx in selected_tokens)
    max_slack = -float("inf")
    for left_columns, left_tokens in left_paths:
        left_required = set(left_tokens) & selected_set
        for right_columns, right_tokens in right_paths:
            required = left_required | (set(right_tokens) & selected_set)
            lhs = positive_token_sum
            for col_idx in (*left_columns, *right_columns):
                lhs += float(edge_coefficients.get(col_idx, 0.0))
            for token_idx in required:
                lhs += min(0.0, float(token_coefficients.get(token_idx, 0.0)))
            max_slack = max(max_slack, lhs - rhs)
    return max_slack


def current_cut_violation(edge_coefficients, token_coefficients, rhs, selected_tokens, f_values, g_values, t_values, num_f):
    lhs = 0.0
    for col_idx, coefficient in edge_coefficients.items():
        if col_idx < num_f:
            lhs += coefficient * float(f_values[col_idx])
        else:
            lhs += coefficient * float(g_values[col_idx - num_f])
    for token_idx in selected_tokens:
        lhs += float(token_coefficients.get(token_idx, 0.0)) * float(t_values[token_idx])
    return lhs - rhs


def feature_summary(records):
    positives = [row for row in records if row.get("has_cut")]
    negatives = [row for row in records if not row.get("has_cut")]
    fields = [
        "score",
        "pair_rows",
        "num_selected_colors",
        "num_shared_colors",
        "num_fractional_union_colors",
        "num_fractional_shared_colors",
        "shared_fractional_mass",
        "union_fractional_mass",
        "shared_candidate_score",
    ]
    return {
        field: {
            "all": quantiles([row[field] for row in records]),
            "cuts": quantiles([row[field] for row in positives]),
            "no_cuts": quantiles([row[field] for row in negatives]),
        }
        for field in fields
    }


def heuristic_tables(records):
    positives = [row for row in records if row.get("has_cut")]
    num_cuts = len(positives)
    lines = []
    for field, reverse in [
        ("score", True),
        ("shared_candidate_score", True),
        ("num_fractional_shared_colors", True),
        ("shared_fractional_mass", True),
        ("pair_rows", False),
    ]:
        ordered = sorted(records, key=lambda row: row[field], reverse=reverse)
        for keep_fraction in (0.1, 0.25, 0.5):
            keep = max(1, int(round(len(ordered) * keep_fraction)))
            kept = ordered[:keep]
            kept_cuts = sum(1 for row in kept if row.get("has_cut"))
            lines.append(
                {
                    "heuristic": f"top_{field}" if reverse else f"bottom_{field}",
                    "keep_fraction": keep_fraction,
                    "kept": keep,
                    "cuts": kept_cuts,
                    "recall": kept_cuts / max(1, num_cuts),
                    "precision": kept_cuts / max(1, keep),
                }
            )
    return lines


def dimension_reduction_summary(records):
    counters = defaultdict(lambda: Counter())
    numeric = defaultdict(lambda: defaultdict(list))
    for row in records:
        for variant in row.get("dimension_variants", []):
            name = variant["name"]
            counters[name]["tested"] += 1
            if variant.get("has_cut"):
                counters[name]["has_cut"] += 1
            if variant.get("full_valid"):
                counters[name]["full_valid"] += 1
            if variant.get("has_cut") and variant.get("full_valid") and variant.get("current_violation", 0.0) > 1e-6:
                counters[name]["valid_cut"] += 1
            for key in ("tokens", "rows_built", "current_violation", "max_full_row_slack", "wall_seconds"):
                if key in variant:
                    numeric[name][key].append(variant[key])
    return {
        name: {
            **dict(counts),
            "metrics": {metric: quantiles(values) for metric, values in numeric[name].items()},
        }
        for name, counts in counters.items()
    }


def examples(records, count):
    positives = sorted([row for row in records if row.get("has_cut")], key=lambda row: row["violation"], reverse=True)
    negatives = sorted([row for row in records if not row.get("has_cut")], key=lambda row: row["score"], reverse=True)
    return {
        "cuts_by_violation": strip_large_fields(positives[:count]),
        "no_cuts_by_score": strip_large_fields(negatives[:count]),
    }


def strip_large_fields(rows):
    output = []
    for row in rows:
        item = dict(row)
        item.pop("selected_tokens", None)
        item.pop("cut", None)
        output.append(item)
    return output


def quantiles(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def format_log(report):
    summary = report["summary"]
    lines = [
        f"checked_tasks={summary['checked_tasks']} next_pair_cuts={summary['next_pair_cuts']} "
        f"active_cuts={summary['active_cuts']} elapsed={summary['elapsed_seconds']:.3f}s",
        f"candidate_pairs={summary['candidate_pairs']} prepared_tasks={summary['prepared_tasks']} "
        f"skipped={summary['prepare_skipped']}",
        "heuristics:",
    ]
    for row in report["heuristics"]:
        lines.append(
            f"  {row['heuristic']} keep={row['keep_fraction']:.0%}: "
            f"recall={row['recall']:.3f} precision={row['precision']:.3f} cuts={row['cuts']}/{row['kept']}"
        )
    lines.append("dimension_reduction:")
    for name, row in report["dimension_reduction"].items():
        metrics = row.get("metrics", {})
        rows = metrics.get("rows_built", {})
        tokens = metrics.get("tokens", {})
        lines.append(
            f"  {name}: tested={row.get('tested', 0)} valid_cut={row.get('valid_cut', 0)} "
            f"full_valid={row.get('full_valid', 0)} rows_p50={rows.get('p50') if rows else None} "
            f"tokens_p50={tokens.get('p50') if tokens else None}"
        )
    return lines


def jsonify_key(value):
    if isinstance(value, tuple):
        return [jsonify_key(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def serializable(value):
    if isinstance(value, tuple):
        return [serializable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


if __name__ == "__main__":
    main()
