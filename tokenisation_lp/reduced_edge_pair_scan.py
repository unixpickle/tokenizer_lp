from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
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
    prepare_lp_data,
    reorder_short_word_pair_candidates,
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
    cuts_path = output_dir / "reduced_edge_pair_cuts.jsonl"
    summary_path = output_dir / "summary.json"
    log_path = output_dir / "summary.log"

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
    tasks, prep_summary = prepare_tasks(args, lp, f_values, t_values, checkpoint["existing_cut_keys"])
    prep_seconds = time.monotonic() - prep_start
    paths_by_word = prep_summary.pop("_paths_by_word")
    LOGGER.info("Prepared tasks=%d prep=%.1fs summary=%s", len(tasks), prep_seconds, prep_summary)

    worker_state = {
        "lp": lp,
        "f_values": f_values,
        "g_values": g_values,
        "t_values": t_values,
        "paths_by_word": paths_by_word,
        "words": words,
        "tolerance": args.cut_tolerance,
        "reduced_row_limit": args.reduced_row_limit,
    }
    worker_count = max(1, int(args.workers))
    batch_size = max(1, int(args.batch_size))
    start = time.monotonic()
    deadline = start + args.time_limit_seconds if args.time_limit_seconds > 0 else None
    checked = 0
    cuts = 0
    skipped = Counter()
    metrics = {
        "build_seconds": 0.0,
        "solve_seconds": 0.0,
        "validation_seconds": 0.0,
        "edge_vars": [],
        "dedup_rows": [],
        "original_rows": [],
    }

    with cuts_path.open("w", encoding="utf-8") as cuts_file:
        if worker_count == 1:
            init_worker(worker_state)
            for task in tasks:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                result = scan_batch_worker([task])[0]
                checked, cuts = handle_result(
                    result,
                    cuts_file,
                    checked,
                    cuts,
                    skipped,
                    metrics,
                    start,
                    len(tasks),
                    args.progress_interval,
                )
        else:
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
                max_in_flight = max(worker_count * 3, 1)

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
                            for result in future.result():
                                checked, cuts = handle_result(
                                    result,
                                    cuts_file,
                                    checked,
                                    cuts,
                                    skipped,
                                    metrics,
                                    start,
                                    len(tasks),
                                    args.progress_interval,
                                )
                            submit_more()
                            break
                    except TimeoutError:
                        pass
                    if deadline is not None and time.monotonic() >= deadline:
                        for future in in_flight:
                            future.cancel()
                        break

    elapsed = time.monotonic() - start
    summary = {
        "run_dir": str(run_dir),
        "candidate_pairs": prep_summary["candidate_pairs"],
        "prepared_tasks": len(tasks),
        "checked": checked,
        "cuts": cuts,
        "cut_rate": cuts / max(1, checked),
        "prepare_seconds": prep_seconds,
        "elapsed_seconds": elapsed,
        "pairs_per_second": checked / elapsed if elapsed else 0.0,
        "workers": worker_count,
        "batch_size": batch_size,
        "reduced_row_limit": args.reduced_row_limit,
        "skipped_prepare": prep_summary["skipped"],
        "skipped_scan": dict(skipped),
        "path_patterns": prep_summary["path_patterns"],
        "metrics": summarize_metrics(metrics),
        "cuts_path": str(cuts_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = format_summary(summary)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        LOGGER.info(line)
    LOGGER.info("Wrote cuts=%s summary=%s log=%s", cuts_path, summary_path, log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan short-word pair cuts with shared fractional colors and fractional-current edge coefficients."
    )
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
    parser.add_argument("--pair-candidate-word-multiplier", type=float, default=4.0)
    parser.add_argument("--pair-candidate-top-words-multiplier", type=float, default=4.0)
    parser.add_argument("--pair-candidate-strategy", choices=("score", "mixed", "random"), default="score")
    parser.add_argument("--pair-candidate-random-seed", type=int, default=0)
    parser.add_argument("--pair-min-fractional-shared-colors", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--max-pairs", type=int, default=0, help="Maximum prepared pairs after filtering; 0 means all.")
    parser.add_argument("--reduced-row-limit", type=int, default=50000)
    parser.add_argument("--time-limit-seconds", type=float, default=1800.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=5000)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def load_checkpoint_state(state_dir: Path) -> dict:
    payload = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    payload["existing_cut_keys"] = {
        tupleify_json_key(key)
        for key in payload.get("existing_cut_keys", [])
    }
    return payload


def prepare_tasks(args, lp, f_values, t_values, existing_cut_keys):
    candidate_max_words = max(
        args.pair_max_words,
        int(math.ceil(args.pair_max_words * max(1.0, args.pair_candidate_word_multiplier))),
    )
    candidate_top_words_per_color = max(
        args.pair_top_words_per_color,
        int(math.ceil(args.pair_top_words_per_color * max(1.0, args.pair_candidate_top_words_multiplier))),
    )
    pair_rows, _ = short_word_pair_candidates(
        lp,
        f_values,
        t_values,
        max_words=candidate_max_words,
        max_word_length=args.pair_max_word_length,
        top_words_per_color=candidate_top_words_per_color,
        tolerance=args.cut_tolerance,
    )
    if args.pair_candidate_strategy == "mixed":
        pair_rows = reorder_short_word_pair_candidates(
            pair_rows,
            lp,
            t_values,
            max_pairs=len(pair_rows),
            tolerance=args.cut_tolerance,
            strategy="mixed",
            random_seed=args.pair_candidate_random_seed,
        )
    elif args.pair_candidate_strategy == "random":
        rng = random.Random(args.pair_candidate_random_seed)
        pair_rows = list(pair_rows)
        rng.shuffle(pair_rows)

    fractional_colors = set(np.flatnonzero((t_values > args.cut_tolerance) & (t_values < 1.0 - args.cut_tolerance)))
    colors_by_word = {}
    paths_by_word = {}
    path_pattern_cache = {}
    tasks = []
    skipped = Counter()
    max_pairs = int(args.max_pairs)

    for _, left_word, right_word in pair_rows:
        if max_pairs > 0 and len(tasks) >= max_pairs:
            break
        if any(key[:3] == ("short_word_pair_hull", left_word, right_word) for key in existing_cut_keys):
            skipped["existing"] += 1
            continue
        left_colors = colors_by_word.setdefault(left_word, set(all_word_token_colors(lp, left_word)))
        right_colors = colors_by_word.setdefault(right_word, set(all_word_token_colors(lp, right_word)))
        selected_tokens = tuple(sorted((left_colors & right_colors) & fractional_colors))
        if len(selected_tokens) < args.pair_min_fractional_shared_colors:
            skipped["shared"] += 1
            continue
        if left_word not in paths_by_word:
            paths_by_word[left_word] = enumerate_word_edge_paths_by_pattern(
                lp,
                left_word,
                max_paths=args.max_paths,
                cache=path_pattern_cache,
            )
        if right_word not in paths_by_word:
            paths_by_word[right_word] = enumerate_word_edge_paths_by_pattern(
                lp,
                right_word,
                max_paths=args.max_paths,
                cache=path_pattern_cache,
            )
        if paths_by_word[left_word] is None or paths_by_word[right_word] is None:
            skipped["paths"] += 1
            continue
        tasks.append((left_word, right_word, selected_tokens))

    return tasks, {
        "candidate_pairs": len(pair_rows),
        "skipped": dict(skipped),
        "path_patterns": len(path_pattern_cache),
        "_paths_by_word": paths_by_word,
    }


WORKER_STATE = {}


def init_worker(state):
    global WORKER_STATE
    WORKER_STATE = state


def scan_batch_worker(tasks):
    return [scan_worker(task) for task in tasks]


def scan_worker(task):
    state = WORKER_STATE
    lp = state["lp"]
    left_word, right_word, selected_tokens = task
    left_paths = state["paths_by_word"][left_word]
    right_paths = state["paths_by_word"][right_word]
    cut = solve_reduced_pair_cut(
        lp,
        state["f_values"],
        state["g_values"],
        state["t_values"],
        left_word,
        right_word,
        selected_tokens,
        left_paths,
        right_paths,
        tolerance=state["tolerance"],
        reduced_row_limit=state["reduced_row_limit"],
    )
    if cut["cut"] is not None:
        cut["cut_record"]["left_word"] = state["words"][left_word]
        cut["cut_record"]["right_word"] = state["words"][right_word]
    return cut


def solve_reduced_pair_cut(
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
    tolerance,
    reduced_row_limit,
):
    start = time.monotonic()
    num_f = lp["num_nonfree_edges"]
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    selected_set = set(selected_tokens)
    word_columns = []
    current_values = []
    edge_meta = Counter()
    for word_idx in (left_word, right_word):
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            value = float(f_values[edge_idx])
            if not (tolerance < value < 1.0 - tolerance):
                continue
            word_columns.append(edge_idx)
            current_values.append(value)
            edge_meta["nonfree"] += 1
            edge_meta["fractional"] += 1
            if lp["nonfree_edge_info"][edge_idx]["token_index"] in selected_set:
                edge_meta["selected_color"] += 1
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            value = float(g_values[edge_idx])
            if not (tolerance < value < 1.0 - tolerance):
                continue
            word_columns.append(num_f + edge_idx)
            current_values.append(value)
            edge_meta["free"] += 1
            edge_meta["fractional"] += 1

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
    seen_rows = set()
    limit_hit = False
    for left_columns, left_tokens in left_paths:
        left_required = set(left_tokens) & selected_set
        left_kept = tuple(sorted(col_position[col_idx] for col_idx in left_columns if col_idx in col_position))
        for right_columns, right_tokens in right_paths:
            required = left_required | (set(right_tokens) & selected_set)
            token_mask = 0
            for token_idx in required:
                token_mask |= 1 << selected_position[token_idx]
            kept_positions = left_kept + tuple(
                sorted(col_position[col_idx] for col_idx in right_columns if col_idx in col_position)
            )
            row_key = (kept_positions, token_mask)
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            if reduced_row_limit > 0 and row_idx >= reduced_row_limit:
                limit_hit = True
                break
            for pos in kept_positions:
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
        if limit_hit:
            break

    build_seconds = time.monotonic() - start
    base = {
        "left_word_idx": int(left_word),
        "right_word_idx": int(right_word),
        "selected_tokens": tuple(selected_tokens),
        "num_selected_tokens": len(selected_tokens),
        "edge_vars": num_edge_vars,
        "dedup_rows": row_idx,
        "original_rows": len(left_paths) * len(right_paths),
        "edge_meta": dict(edge_meta),
        "build_seconds": build_seconds,
        "solve_seconds": 0.0,
        "validation_seconds": 0.0,
        "skip_reason": None,
        "cut": None,
        "cut_record": None,
    }
    if limit_hit:
        base["skip_reason"] = "reduced_rows"
        return base

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
    solve_start = time.monotonic()
    result = linprog(c=objective, A_ub=constraints, b_ub=np.array(rhs, dtype=float), bounds=bounds, method="highs")
    base["solve_seconds"] = time.monotonic() - solve_start
    if not result.success:
        base["skip_reason"] = "linprog"
        base["message"] = result.message
        return base
    violation = -float(result.fun)
    if violation <= tolerance:
        return base

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
    rhs_value = -float(result.x[gamma_col])
    validation_start = time.monotonic()
    max_slack = max_full_row_slack(edge_coefficients, token_coefficients, rhs_value, selected_tokens, left_paths, right_paths)
    current_violation = current_cut_violation(
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
        base["max_full_row_slack"] = float(max_slack)
        base["current_violation"] = float(current_violation)
        return base

    t_offset = lp["num_nonfree_edges"] + lp["num_free_edges"]
    entries = [(int(col_idx), float(coefficient)) for col_idx, coefficient in edge_coefficients.items()]
    entries.extend(
        (int(t_offset + token_idx), float(coefficient))
        for token_idx, coefficient in token_coefficients.items()
    )
    coefficient_key = (
        round(-rhs_value, 8),
        tuple(round(float(token_coefficients.get(token_idx, 0.0)), 8) for token_idx in selected_tokens),
        tuple(sorted((int(col_idx), round(float(coefficient), 8)) for col_idx, coefficient in edge_coefficients.items())),
    )
    key = ("short_word_pair_hull_reduced_fractional_edges", left_word, right_word, tuple(selected_tokens), coefficient_key)
    base["cut"] = (float(current_violation), key, entries, float(rhs_value))
    base["cut_record"] = {
        "violation": float(current_violation),
        "reduced_objective_violation": float(violation),
        "left_word_idx": int(left_word),
        "right_word_idx": int(right_word),
        "num_shared_fractional_colors": len(selected_tokens),
        "edge_vars": num_edge_vars,
        "dedup_rows": row_idx,
        "original_rows": len(left_paths) * len(right_paths),
        "rhs": float(rhs_value),
        "num_edge_coefficients": len(edge_coefficients),
        "num_token_coefficients": len(token_coefficients),
        "max_full_row_slack": float(max_slack),
        "entries": entries,
        "key": key,
    }
    return base


def max_full_row_slack(edge_coefficients, token_coefficients, rhs, selected_tokens, left_paths, right_paths):
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


def handle_result(result, cuts_file, checked, cuts, skipped, metrics, start, total, progress_interval):
    checked += 1
    reason = result.get("skip_reason")
    if reason:
        skipped[reason] += 1
    metrics["build_seconds"] += float(result.get("build_seconds", 0.0))
    metrics["solve_seconds"] += float(result.get("solve_seconds", 0.0))
    metrics["validation_seconds"] += float(result.get("validation_seconds", 0.0))
    metrics["edge_vars"].append(int(result.get("edge_vars", 0)))
    metrics["dedup_rows"].append(int(result.get("dedup_rows", 0)))
    metrics["original_rows"].append(int(result.get("original_rows", 0)))
    if result.get("cut") is not None:
        cuts += 1
        record = dict(result["cut_record"])
        record["rank"] = cuts
        cuts_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        cuts_file.flush()
    if checked == 1 or checked % progress_interval == 0 or checked == total:
        elapsed = time.monotonic() - start
        LOGGER.info(
            "progress checked=%d/%d cuts=%d skipped=%s elapsed=%.1fs rate=%.1f/s",
            checked,
            total,
            cuts,
            dict(skipped),
            elapsed,
            checked / elapsed if elapsed else 0.0,
        )
    return checked, cuts


def summarize_metrics(metrics):
    return {
        "build_seconds": metrics["build_seconds"],
        "solve_seconds": metrics["solve_seconds"],
        "validation_seconds": metrics["validation_seconds"],
        "edge_vars": quantiles(metrics["edge_vars"]),
        "dedup_rows": quantiles(metrics["dedup_rows"]),
        "original_rows": quantiles(metrics["original_rows"]),
    }


def quantiles(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def format_summary(summary):
    return [
        (
            "reduced-edge pair scan: "
            f"checked={summary['checked']} cuts={summary['cuts']} "
            f"cut_rate={summary['cut_rate']:.4f} elapsed={summary['elapsed_seconds']:.1f}s "
            f"rate={summary['pairs_per_second']:.1f}/s"
        ),
        (
            f"candidate_pairs={summary['candidate_pairs']} prepared={summary['prepared_tasks']} "
            f"prep={summary['prepare_seconds']:.1f}s skipped_prepare={summary['skipped_prepare']}"
        ),
        (
            f"skipped_scan={summary['skipped_scan']} reduced_row_limit={summary['reduced_row_limit']} "
            f"workers={summary['workers']} batch_size={summary['batch_size']}"
        ),
        f"metrics={summary['metrics']}",
        f"cuts_path={summary['cuts_path']}",
    ]


if __name__ == "__main__":
    main()
