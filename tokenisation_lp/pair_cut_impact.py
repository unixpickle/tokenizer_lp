from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from tokenisation_lp.alternative_pair_diagnostics import (
    build_word_profiles,
    merge_candidates,
    propose_candidates,
)
from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    HighsWarmLpSolver,
    build_standard_form,
    chunked,
    count_pretokenized_strings,
    init_pair_hull_worker,
    pair_hull_batch_worker,
    prepare_lp_data,
    resolve_pair_hull_workers,
    short_word_pair_candidates,
)
from tokenisation_lp.pair_hull_diagnostics import (
    apply_individual_hulls,
    prepare_pair_tasks,
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
    # Keep a live HiGHS model and basis. Do not use solution-cache shortcuts here.
    solver = HighsWarmLpSolver(
        c=lp["c"],
        A_ub=lp["A_ub"],
        b_ub=b_ub,
        A_eq=lp["A_eq"],
        b_eq=lp["b_eq"],
        lb=lp["lb"],
        ub=lp["ub"],
        cache_dir=None,
    )
    solution = timed_solve(solver, "root")
    if args.apply_individual_hulls:
        solution = apply_individual_hulls(args, lp, solver, solution)
    if not solution.success:
        raise RuntimeError(f"root LP failed: {solution.message}")

    root_bound = float(solution.fun)
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    f_values = solution.x[:num_f]
    g_values = solution.x[num_f : num_f + num_g]
    t_values = solution.x[num_f + num_g :]
    LOGGER.info("Singleton impact root bound %.6f", root_bound)

    pair_rows, metadata = candidate_pairs(args, lp, words, freqs, f_values, t_values)
    tasks, paths_by_word, skipped, pattern_count = prepare_pair_tasks(
        lp,
        pair_rows,
        max_pairs=len(pair_rows),
        max_colors=args.max_colors,
        max_pair_rows=args.max_pair_rows,
        max_paths=args.max_paths,
    )
    LOGGER.info(
        "Prepared singleton-impact pair tasks: candidates=%d tasks=%d skipped=%s patterns=%d",
        len(pair_rows),
        len(tasks),
        dict(skipped),
        pattern_count,
    )
    cuts = find_pair_cuts(
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
    if args.max_cuts and len(cuts) > args.max_cuts:
        cuts = sorted(cuts, key=lambda row: row["violation"], reverse=True)[: args.max_cuts]
    LOGGER.info("Evaluating singleton bound impact for %d violated pair cuts", len(cuts))

    impact_rows = evaluate_singleton_impacts(
        solver,
        lp,
        cuts,
        root_bound,
        workers=args.impact_workers,
        batch_size=args.impact_batch_size,
    )
    for row in impact_rows[: args.print_top]:
        key = pair_key(row["left_word"], row["right_word"])
        LOGGER.info(
            "impact delta=%.6f violation=%.6f pair=%r/%r strategies=%s solve=%.3fs",
            row["bound_delta"],
            row["violation"],
            words[row["left_word"]],
            words[row["right_word"]],
            ",".join(sorted(metadata.get(key, {}).get("strategies", []))) or args.candidate_source,
            row["solve_seconds"],
        )

    report = {
        "run": {
            "data_dir": str(Path(args.data_dir).expanduser()),
            "vocab_size": args.vocab_size,
            "pretokenizer": args.pretokenizer,
            "candidate_source": args.candidate_source,
            "root_bound": root_bound,
            "num_candidates": len(pair_rows),
            "num_tasks": len(tasks),
            "num_cuts": len(cuts),
            "skipped": dict(skipped),
        },
        "summary": summarize_impacts(impact_rows),
        "rows": [
            {
                **row,
                "left_word_text": words[row["left_word"]],
                "right_word_text": words[row["right_word"]],
                "strategies": sorted(metadata.get(pair_key(row["left_word"], row["right_word"]), {}).get("strategies", [])),
            }
            for row in impact_rows
        ],
    }
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote singleton impact report: %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure LP-bound lift from adding each pair-hull cut by itself.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--apply-individual-hulls", action="store_true")
    parser.add_argument("--individual-rounds", type=int, default=3)
    parser.add_argument("--individual-max-words", type=int, default=12000)
    parser.add_argument("--individual-max-length", type=int, default=12)
    parser.add_argument("--individual-max-colors", type=int, default=96)
    parser.add_argument("--candidate-source", default="alternative", choices=("baseline", "alternative"))
    parser.add_argument("--max-words", type=int, default=700)
    parser.add_argument("--max-word-length", type=int, default=12)
    parser.add_argument("--max-colors", type=int, default=96)
    parser.add_argument("--max-pair-rows", type=int, default=250000)
    parser.add_argument("--max-pairs", type=int, default=8000)
    parser.add_argument("--top-words-per-color", type=int, default=36)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cut-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-candidates-per-strategy", type=int, default=1000)
    parser.add_argument("--max-total-pairs", type=int, default=5000)
    parser.add_argument("--neighbors-per-word", type=int, default=10)
    parser.add_argument("--top-colors-per-word", type=int, default=10)
    parser.add_argument("--max-cuts", type=int, default=0)
    parser.add_argument("--impact-workers", type=int, default=0)
    parser.add_argument("--impact-batch-size", type=int, default=1)
    parser.add_argument("--print-top", type=int, default=20)
    parser.add_argument("--output", default="/tmp/tokenizer_lp_pair_cut_impact.json")
    # apply_individual_hulls expects this arg name.
    parser.set_defaults(lp_solution_cache_dir=None)
    return parser.parse_args()


def timed_solve(solver, label: str):
    start = time.monotonic()
    solution = solver.solve()
    LOGGER.info(
        "%s solve status=%s objective=%.6f elapsed=%.3fs",
        label,
        solution.message,
        float(solution.fun) if solution.success else float("nan"),
        time.monotonic() - start,
    )
    return solution


def candidate_pairs(args, lp, words, freqs, f_values, t_values):
    if args.candidate_source == "baseline":
        rows, _ = short_word_pair_candidates(
            lp,
            f_values,
            t_values,
            max_words=args.max_words,
            max_word_length=args.max_word_length,
            top_words_per_color=args.top_words_per_color,
            tolerance=args.cut_tolerance,
        )
        rows = rows[: args.max_pairs]
        metadata = {
            pair_key(left, right): {"strategies": {"baseline_shared_color"}, "strategy_scores": {"baseline_shared_color": score}}
            for score, left, right in rows
        }
        return rows, metadata

    profiles = build_word_profiles(lp, words, freqs, f_values, t_values, args)
    strategies = propose_candidates(lp, f_values, t_values, profiles, args)
    rows, metadata = merge_candidates(strategies, args.max_total_pairs)
    return rows, metadata


def find_pair_cuts(lp, f_values, g_values, t_values, paths_by_word, tasks, *, tolerance, workers, batch_size):
    worker_state = {
        "lp": lp,
        "f_values": f_values,
        "g_values": g_values,
        "t_values": t_values,
        "paths_by_word": paths_by_word,
        "tolerance": tolerance,
    }
    tuple_tasks = [
        (task["left_word"], task["right_word"], task["selected_tokens"])
        for task in tasks
    ]
    worker_count = resolve_pair_hull_workers(workers)
    start = time.monotonic()
    results = []
    if worker_count == 1:
        init_pair_hull_worker(worker_state)
        for batch in chunked(tuple_tasks, batch_size):
            results.extend(pair_hull_batch_worker(batch))
    else:
        context = mp.get_context("fork") if hasattr(os, "fork") else None
        batches = list(chunked(tuple_tasks, batch_size))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=init_pair_hull_worker,
            initargs=(worker_state,),
        ) as executor:
            futures = [executor.submit(pair_hull_batch_worker, batch) for batch in batches]
            for future in as_completed(futures):
                results.extend(future.result())
    cuts = []
    for result in results:
        cut = result["cut"]
        if cut is None:
            continue
        violation, full_key, entries, rhs = cut
        _, left_word, right_word, selected_tokens, coefficient_key = full_key
        cuts.append(
            {
                "violation": float(violation),
                "left_word": int(left_word),
                "right_word": int(right_word),
                "selected_tokens": tuple(int(token_idx) for token_idx in selected_tokens),
                "coefficient_key": coefficient_key,
                "entries": entries,
                "rhs": float(rhs),
                "separator_build_seconds": float(result["build_seconds"]),
                "separator_solve_seconds": float(result["solve_seconds"]),
            }
        )
    cuts.sort(key=lambda row: row["violation"], reverse=True)
    LOGGER.info(
        "Found pair cuts: checked=%d cuts=%d workers=%d elapsed=%.3fs",
        len(tuple_tasks),
        len(cuts),
        worker_count,
        time.monotonic() - start,
    )
    return cuts


def evaluate_singleton_impacts(solver, lp, cuts, root_bound, *, workers: int, batch_size: int):
    if not cuts:
        return []
    matrix, rhs = cut_matrix(cuts, len(lp["c"]))
    inactive_rhs = np.full(len(cuts), solver.highspy.kHighsInf, dtype=float)
    start_row = solver.add_ub_rows(matrix, inactive_rhs)
    if start_row is None:
        return []

    # Re-solve once with inactive rows so HiGHS extends the basis to the added rows.
    inactive_solution = timed_solve(solver, "inactive singleton rows")
    if not inactive_solution.success:
        raise RuntimeError(f"inactive-row LP failed: {inactive_solution.message}")

    with tempfile.TemporaryDirectory(prefix="tokenizer_lp_basis_") as tmp_dir:
        basis_path = str(Path(tmp_dir) / "inactive_pair_cuts.bas")
        status = solver.highs.writeBasis(basis_path)
        LOGGER.info("Wrote inactive singleton basis %s status=%s", basis_path, status)
        rows = evaluate_singleton_impacts_parallel(
            solver,
            cuts,
            rhs,
            root_bound,
            start_row=start_row,
            basis_path=basis_path,
            workers=workers,
            batch_size=batch_size,
        )
    rows.sort(key=lambda row: (row["bound_delta"], row["violation"]), reverse=True)
    return rows


def evaluate_singleton_impacts_parallel(
    solver,
    cuts,
    rhs,
    root_bound,
    *,
    start_row: int,
    basis_path: str,
    workers: int,
    batch_size: int,
):
    tasks = [
        {
            "cut_index": idx,
            "cut": cut,
            "rhs": float(rhs[idx]),
            "row_idx": start_row + idx,
        }
        for idx, cut in enumerate(cuts)
    ]
    worker_count = resolve_pair_hull_workers(workers)
    worker_state = {
        "c": solver.c,
        "A_ub": solver.A_ub,
        "b_ub": solver.b_ub,
        "A_eq": solver.A_eq,
        "b_eq": solver.b_eq,
        "lb": solver.lb,
        "ub": solver.ub,
        "basis_path": basis_path,
        "root_bound": root_bound,
    }
    start = time.monotonic()
    if worker_count == 1:
        init_impact_worker(worker_state)
        rows = []
        for batch in chunked(tasks, batch_size):
            rows.extend(pair_cut_impact_batch_worker(batch))
    else:
        solver.highs.resetGlobalScheduler(True)
        context = mp.get_context("fork") if hasattr(os, "fork") else None
        batches = list(chunked(tasks, batch_size))
        rows = []
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=init_impact_worker,
            initargs=(worker_state,),
        ) as executor:
            futures = [executor.submit(pair_cut_impact_batch_worker, batch) for batch in batches]
            for future in as_completed(futures):
                rows.extend(future.result())
    LOGGER.info(
        "Singleton impact subsolves done: cuts=%d workers=%d batch_size=%d elapsed=%.3fs",
        len(tasks),
        worker_count,
        max(1, int(batch_size)),
        time.monotonic() - start,
    )
    return rows


IMPACT_WORKER_STATE = {}


def init_impact_worker(state):
    global IMPACT_WORKER_STATE
    IMPACT_WORKER_STATE = state


def pair_cut_impact_batch_worker(tasks):
    state = IMPACT_WORKER_STATE
    solver = HighsWarmLpSolver(
        c=state["c"],
        A_ub=state["A_ub"],
        b_ub=state["b_ub"],
        A_eq=state["A_eq"],
        b_eq=state["b_eq"],
        lb=state["lb"],
        ub=state["ub"],
        cache_dir=None,
        highs_threads=1,
        highs_parallel="off",
    )
    solver.highs.setOptionValue("presolve", "off")
    solver.has_basis = True
    return [pair_cut_impact_worker(solver, task, state["basis_path"], state["root_bound"]) for task in tasks]


def pair_cut_impact_worker(solver, task, basis_path: str, root_bound: float):
    cut = task["cut"]
    lower = -solver.highspy.kHighsInf
    upper_inactive = solver.highspy.kHighsInf
    solver.highs.readBasis(basis_path)
    solver.change_ub_row_bounds(task["row_idx"], lower, task["rhs"])
    start = time.monotonic()
    solution = solver.solve()
    solve_seconds = time.monotonic() - start
    if not solution.success:
        bound = float("nan")
        delta = float("nan")
        message = solution.message
    else:
        bound = float(solution.fun)
        delta = bound - root_bound
        message = solution.message
    solver.change_ub_row_bounds(task["row_idx"], lower, upper_inactive)
    return {
        "left_word": cut["left_word"],
        "right_word": cut["right_word"],
        "violation": cut["violation"],
        "bound": bound,
        "bound_delta": delta,
        "rhs": cut["rhs"],
        "num_entries": len(cut["entries"]),
        "separator_build_seconds": cut["separator_build_seconds"],
        "separator_solve_seconds": cut["separator_solve_seconds"],
        "solve_seconds": solve_seconds,
        "status": message,
        "cut_index": task["cut_index"],
    }


def cut_matrix(cuts, num_vars: int):
    rows = []
    cols = []
    data = []
    rhs = []
    for row_idx, cut in enumerate(cuts):
        rhs.append(cut["rhs"])
        for col_idx, coefficient in cut["entries"]:
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(float(coefficient))
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(len(cuts), num_vars), dtype=float).tocsr()
    return matrix, np.array(rhs, dtype=float)


def summarize_impacts(rows):
    if not rows:
        return {}
    deltas = np.array([row["bound_delta"] for row in rows], dtype=float)
    violations = np.array([row["violation"] for row in rows], dtype=float)
    return {
        "count": len(rows),
        "positive_delta_count": int(np.sum(deltas > 1e-8)),
        "max_delta": float(np.max(deltas)),
        "median_delta": float(np.median(deltas)),
        "sum_singleton_deltas": float(np.sum(deltas)),
        "max_violation": float(np.max(violations)),
        "corr_violation_delta": float(np.corrcoef(violations, deltas)[0, 1]) if len(rows) >= 2 else 0.0,
    }


def pair_key(left, right):
    return tuple(sorted((int(left), int(right))))


if __name__ == "__main__":
    main()
