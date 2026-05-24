from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    HighsWarmLpSolver,
    all_word_token_colors,
    build_standard_form,
    chunked,
    count_pretokenized_strings,
    enumerate_word_edge_paths_by_pattern,
    pair_full_upward_hull_cut_from_paths,
    prepare_lp_data,
    resolve_pair_hull_workers,
    separate_short_word_full_hull_cut_specs,
    short_word_pair_candidates,
)
from tokenisation_lp.pretokenization import (
    DEFAULT_SPECIAL_TOKENS,
    build_pretokenizer,
    byte_level_alphabet,
)


LOGGER = logging.getLogger(__name__)
PAIR_DIAGNOSTIC_WORKER_STATE = {}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

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
    budget = args.vocab_size - len(DEFAULT_SPECIAL_TOKENS) - len(byte_level_alphabet())
    if budget <= 0:
        raise ValueError("vocab size leaves no LP token budget")

    b_ub = lp["b_ub"].copy()
    b_ub[lp["budget_row"]] = float(budget)
    solver = HighsWarmLpSolver(
        c=lp["c"],
        A_ub=lp["A_ub"],
        b_ub=b_ub,
        A_eq=lp["A_eq"],
        b_eq=lp["b_eq"],
        lb=lp["lb"],
        ub=lp["ub"],
        cache_dir=args.lp_solution_cache_dir,
    )
    solution = solver.solve()
    if not solution.success:
        raise RuntimeError(f"LP solve failed: {solution.message}")
    LOGGER.info("Base LP bound %.3f", solution.fun)
    if args.apply_individual_hulls:
        solution = apply_individual_hulls(args, lp, solver, solution)

    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    f_values = solution.x[:num_f]
    g_values = solution.x[num_f : num_f + num_g]
    t_values = solution.x[num_f + num_g :]

    start = time.monotonic()
    pair_rows, word_color_scores = short_word_pair_candidates(
        lp,
        f_values,
        t_values,
        max_words=args.max_words,
        max_word_length=args.max_word_length,
        top_words_per_color=args.top_words_per_color,
        tolerance=args.cut_tolerance,
    )
    tasks, paths_by_word, skipped, pattern_count = prepare_pair_tasks(
        lp,
        pair_rows,
        max_pairs=args.max_pairs,
        max_colors=args.max_colors,
        max_pair_rows=args.max_pair_rows,
        max_paths=args.max_paths,
    )
    LOGGER.info(
        "Prepared pair diagnostics: candidates=%d tasks=%d skipped=%s patterns=%d",
        len(pair_rows),
        len(tasks),
        dict(skipped),
        pattern_count,
    )

    results = run_pair_diagnostics(
        lp,
        f_values,
        g_values,
        t_values,
        paths_by_word,
        tasks,
        tolerance=args.cut_tolerance,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    elapsed = time.monotonic() - start
    enriched = [
        enrich_result(row, lp, words, freqs, tokens, f_values, t_values, word_color_scores)
        for row in results
    ]
    report = build_report(enriched, examples_per_class=args.examples_per_class)
    report["run"] = {
        "data_dir": str(Path(args.data_dir).expanduser()),
        "vocab_size": args.vocab_size,
        "pretokenizer": args.pretokenizer,
        "max_pairs": args.max_pairs,
        "workers": resolve_pair_hull_workers(args.workers),
        "batch_size": max(1, args.batch_size),
        "elapsed_seconds": elapsed,
        "lp_bound": float(solution.fun),
        "applied_individual_hulls": bool(args.apply_individual_hulls),
    }
    report["skipped"] = dict(skipped)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    LOGGER.info(
        "Pair diagnostics done: checked=%d cuts=%d elapsed=%.3fs output=%s",
        len(enriched),
        sum(1 for row in enriched if row["has_cut"]),
        elapsed,
        output_path,
    )
    for line in report["heuristics"]:
        LOGGER.info("%s", line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump pair-hull positive/negative examples and heuristic summaries.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8)
    parser.add_argument("--lp-solution-cache-dir", default="/tmp/tokenizer_lp_solution_cache")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--apply-individual-hulls", action="store_true")
    parser.add_argument("--individual-rounds", type=int, default=3)
    parser.add_argument("--individual-max-words", type=int, default=12000)
    parser.add_argument("--individual-max-length", type=int, default=12)
    parser.add_argument("--individual-max-colors", type=int, default=96)
    parser.add_argument("--max-words", type=int, default=500)
    parser.add_argument("--max-word-length", type=int, default=12)
    parser.add_argument("--max-colors", type=int, default=96)
    parser.add_argument("--max-pair-rows", type=int, default=250000)
    parser.add_argument("--max-pairs", type=int, default=8000)
    parser.add_argument("--top-words-per-color", type=int, default=36)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cut-tolerance", type=float, default=1e-6)
    parser.add_argument("--examples-per-class", type=int, default=20)
    parser.add_argument("--output", default="/tmp/tokenizer_lp_pair_hull_diagnostics.json")
    return parser.parse_args()


def apply_individual_hulls(args, lp, solver, solution):
    existing = set()
    current = solution
    num_vars = len(lp["c"])
    for round_idx in range(args.individual_rounds):
        num_f = lp["num_nonfree_edges"]
        num_g = lp["num_free_edges"]
        f_values = current.x[:num_f]
        g_values = current.x[num_f : num_f + num_g]
        t_values = current.x[num_f + num_g :]
        start = time.monotonic()
        cuts = separate_short_word_full_hull_cut_specs(
            lp,
            f_values,
            g_values,
            t_values,
            existing_cut_keys=existing,
            tolerance=args.cut_tolerance,
            max_words=args.individual_max_words,
            max_word_length=args.individual_max_length,
            max_colors=args.individual_max_colors,
            max_paths=args.max_paths,
        )
        LOGGER.info(
            "individual hull round %d cuts=%d max=%.6g sep=%.3fs",
            round_idx,
            len(cuts),
            max((cut[0] for cut in cuts), default=0.0),
            time.monotonic() - start,
        )
        if not cuts:
            break
        matrix, rhs = cut_specs_to_matrix(cuts, num_vars)
        existing.update(cut[1] for cut in cuts)
        solver.add_ub_rows(matrix, rhs)
        current = solver.solve()
        if not current.success:
            raise RuntimeError(f"LP solve failed after cuts: {current.message}")
        LOGGER.info("individual hull round %d bound %.3f", round_idx, current.fun)
    return current


def cut_specs_to_matrix(cuts, num_vars: int):
    rows = []
    cols = []
    data = []
    rhs_values = []
    for row_idx, (_, _, entries, rhs) in enumerate(cuts):
        rhs_values.append(float(rhs))
        for col_idx, coefficient in entries:
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(float(coefficient))
    return sp.coo_matrix((data, (rows, cols)), shape=(len(cuts), num_vars), dtype=float).tocsr(), np.array(rhs_values)


def prepare_pair_tasks(lp, pair_rows, *, max_pairs, max_colors, max_pair_rows, max_paths):
    path_pattern_cache = {}
    paths_by_word = {}
    colors_by_word = {}
    tasks = []
    skipped = defaultdict(int)
    for rank, (score, left_word, right_word) in enumerate(pair_rows[:max_pairs], start=1):
        left_colors = colors_by_word.setdefault(left_word, all_word_token_colors(lp, left_word))
        right_colors = colors_by_word.setdefault(right_word, all_word_token_colors(lp, right_word))
        selected_tokens = tuple(sorted(set(left_colors) | set(right_colors)))
        if len(selected_tokens) < 2 or len(selected_tokens) > max_colors:
            skipped["colors"] += 1
            continue
        if left_word not in paths_by_word:
            paths_by_word[left_word] = enumerate_word_edge_paths_by_pattern(
                lp,
                left_word,
                max_paths=max_paths,
                cache=path_pattern_cache,
            )
        if right_word not in paths_by_word:
            paths_by_word[right_word] = enumerate_word_edge_paths_by_pattern(
                lp,
                right_word,
                max_paths=max_paths,
                cache=path_pattern_cache,
            )
        left_paths = paths_by_word[left_word]
        right_paths = paths_by_word[right_word]
        if left_paths is None or right_paths is None:
            skipped["paths"] += 1
            continue
        pair_rows_count = len(left_paths) * len(right_paths)
        if pair_rows_count > max_pair_rows:
            skipped["rows"] += 1
            continue
        tasks.append(
            {
                "rank": rank,
                "score": float(score),
                "left_word": int(left_word),
                "right_word": int(right_word),
                "selected_tokens": selected_tokens,
                "left_paths": len(left_paths),
                "right_paths": len(right_paths),
                "pair_rows": pair_rows_count,
            }
        )
    return tasks, paths_by_word, skipped, len(path_pattern_cache)


def run_pair_diagnostics(lp, f_values, g_values, t_values, paths_by_word, tasks, *, tolerance, workers, batch_size):
    worker_count = resolve_pair_hull_workers(workers)
    state = {
        "lp": lp,
        "f_values": f_values,
        "g_values": g_values,
        "t_values": t_values,
        "paths_by_word": paths_by_word,
        "tolerance": tolerance,
    }
    if worker_count == 1:
        init_pair_diagnostic_worker(state)
        return [pair_diagnostic_worker(task) for task in tasks]

    batches = list(chunked(tasks, batch_size))
    context = mp.get_context("fork") if hasattr(os, "fork") else None
    results = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=init_pair_diagnostic_worker,
        initargs=(state,),
    ) as executor:
        futures = [executor.submit(pair_diagnostic_batch_worker, batch) for batch in batches]
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda row: row["rank"])
    LOGGER.info(
        "Checked pair diagnostics: tasks=%d batches=%d workers=%d batch_size=%d",
        len(tasks),
        len(batches),
        worker_count,
        max(1, batch_size),
    )
    return results


def init_pair_diagnostic_worker(state):
    global PAIR_DIAGNOSTIC_WORKER_STATE
    PAIR_DIAGNOSTIC_WORKER_STATE = state


def pair_diagnostic_batch_worker(tasks):
    return [pair_diagnostic_worker(task) for task in tasks]


def pair_diagnostic_worker(task):
    state = PAIR_DIAGNOSTIC_WORKER_STATE
    lp = state["lp"]
    left_word = task["left_word"]
    right_word = task["right_word"]
    selected_tokens = task["selected_tokens"]
    cut = pair_full_upward_hull_cut_from_paths(
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
    )
    result = dict(task)
    if cut is None:
        result.update({"violation": 0.0, "has_cut": False, "build_seconds": 0.0, "solve_seconds": 0.0})
        return result
    violation, edge_coefficients, token_coefficients, rhs, _, build_seconds, solve_seconds = cut
    result.update(
        {
            "violation": float(violation),
            "has_cut": True,
            "build_seconds": float(build_seconds),
            "solve_seconds": float(solve_seconds),
            "rhs": float(rhs),
            "num_edge_coefficients": len(edge_coefficients),
            "num_token_coefficients": len(token_coefficients),
        }
    )
    return result


def enrich_result(row, lp, words, freqs, tokens, f_values, t_values, word_color_scores):
    left_word = row["left_word"]
    right_word = row["right_word"]
    left_colors = set(all_word_token_colors(lp, left_word))
    right_colors = set(all_word_token_colors(lp, right_word))
    shared_colors = left_colors & right_colors
    fractional_shared = [
        token_idx
        for token_idx in shared_colors
        if 1e-6 < float(t_values[token_idx]) < 1.0 - 1e-6
    ]
    enriched = dict(row)
    enriched.update(
        {
            "left": word_summary(lp, words, freqs, tokens, f_values, t_values, left_word),
            "right": word_summary(lp, words, freqs, tokens, f_values, t_values, right_word),
            "num_selected_colors": len(row["selected_tokens"]),
            "num_shared_colors": len(shared_colors),
            "num_fractional_shared_colors": len(fractional_shared),
            "shared_fractional_color_mass": float(sum(t_values[token_idx] for token_idx in fractional_shared)),
            "shared_candidate_score": float(
                sum(
                    min(
                        word_color_scores[left_word].get(token_idx, 0.0),
                        word_color_scores[right_word].get(token_idx, 0.0),
                    )
                    for token_idx in fractional_shared
                )
            ),
            "selected_color_values": [
                {
                    "token_index": int(token_idx),
                    "token": tokens[token_idx].token,
                    "value": float(t_values[token_idx]),
                }
                for token_idx in row["selected_tokens"]
                if 1e-6 < float(t_values[token_idx]) < 1.0 - 1e-6
            ][:16],
        }
    )
    return enriched


def word_summary(lp, words, freqs, tokens, f_values, t_values, word_idx):
    edges = []
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        info = lp["nonfree_edge_info"][edge_idx]
        value = float(f_values[edge_idx])
        token_idx = info["token_index"]
        token_value = float(t_values[token_idx])
        if value <= 1e-8 and token_value <= 1e-8:
            continue
        edges.append(
            {
                "token_index": int(token_idx),
                "token": tokens[token_idx].token,
                "start": int(info["start"]),
                "end": int(info["end"]),
                "length": int(info["end"] - info["start"]),
                "edge_value": value,
                "color_value": token_value,
                "fractional_signal": min(value, 1.0 - value) * min(token_value, 1.0 - token_value),
            }
        )
    edges.sort(key=lambda item: (item["fractional_signal"], item["edge_value"], item["length"]), reverse=True)
    return {
        "word_index": int(word_idx),
        "word": words[word_idx],
        "length": int(lp["word_lengths"][word_idx]),
        "frequency": int(freqs[word_idx]),
        "top_edges": edges[:12],
    }


def build_report(rows, *, examples_per_class):
    positives = [row for row in rows if row["has_cut"]]
    negatives = [row for row in rows if not row["has_cut"]]
    positives_by_violation = sorted(positives, key=lambda row: row["violation"], reverse=True)
    negatives_by_score = sorted(negatives, key=lambda row: row["score"], reverse=True)
    negatives_by_rows = sorted(negatives, key=lambda row: row["pair_rows"], reverse=True)
    return {
        "summary": {
            "checked": len(rows),
            "cuts": len(positives),
            "no_cuts": len(negatives),
            "cut_rate": len(positives) / max(1, len(rows)),
            "features": feature_summary(rows, positives, negatives),
        },
        "heuristics": heuristic_summary(rows, positives),
        "examples": {
            "cuts_by_violation": positives_by_violation[:examples_per_class],
            "no_cuts_by_candidate_score": negatives_by_score[:examples_per_class],
            "no_cuts_by_pair_rows": negatives_by_rows[:examples_per_class],
        },
    }


def feature_summary(rows, positives, negatives):
    fields = [
        "score",
        "pair_rows",
        "num_selected_colors",
        "num_shared_colors",
        "num_fractional_shared_colors",
        "shared_fractional_color_mass",
        "shared_candidate_score",
    ]
    return {
        field: {
            "all": quantiles([row[field] for row in rows]),
            "cuts": quantiles([row[field] for row in positives]),
            "no_cuts": quantiles([row[field] for row in negatives]),
        }
        for field in fields
    }


def quantiles(values):
    if not values:
        return None
    array = np.array(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def heuristic_summary(rows, positives):
    if not rows:
        return []
    num_cuts = len(positives)
    lines = []
    for field, reverse in [
        ("score", True),
        ("shared_candidate_score", True),
        ("num_fractional_shared_colors", True),
        ("pair_rows", False),
    ]:
        ordered = sorted(rows, key=lambda row: row[field], reverse=reverse)
        for keep_fraction in (0.1, 0.25, 0.5):
            keep = max(1, int(round(len(ordered) * keep_fraction)))
            kept = ordered[:keep]
            kept_cuts = sum(1 for row in kept if row["has_cut"])
            recall = kept_cuts / max(1, num_cuts)
            precision = kept_cuts / max(1, keep)
            lines.append(
                f"keep top {keep_fraction:.0%} by {field}: recall={recall:.3f} "
                f"precision={precision:.3f} kept={keep} cuts={kept_cuts}/{num_cuts}"
            )
    return lines


if __name__ == "__main__":
    main()
