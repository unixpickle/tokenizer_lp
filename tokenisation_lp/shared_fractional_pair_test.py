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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    all_word_token_colors,
    build_standard_form,
    chunked,
    count_pretokenized_strings,
    enumerate_word_edge_paths_by_pattern,
    init_pair_hull_worker,
    pair_hull_batch_worker,
    prepare_lp_data,
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
    cuts_path = output_dir / "shared_fractional_pair_cuts.jsonl"
    summary_path = output_dir / "summary.json"
    text_log_path = output_dir / "summary.log"

    LOGGER.info("Loading final LP state from %s", state_dir)
    checkpoint = load_checkpoint_state(state_dir)
    existing_cut_keys = checkpoint["existing_cut_keys"]
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
    expected_vars = num_f + num_g + lp["num_tokens"]
    if x_values.shape != (expected_vars,):
        raise ValueError(f"latest_solution shape {x_values.shape} does not match LP vars {expected_vars}")
    f_values = x_values[:num_f]
    g_values = x_values[num_f : num_f + num_g]
    t_values = x_values[num_f + num_g :]

    pair_rows = build_pair_rows(args, lp, f_values, t_values)
    LOGGER.info("Built candidate pair pool: %d", len(pair_rows))

    prepare_start = time.monotonic()
    tasks, task_metadata, skipped, path_patterns = prepare_random_tasks(
        args,
        lp,
        pair_rows,
        t_values,
        existing_cut_keys,
    )
    prepare_seconds = time.monotonic() - prepare_start
    LOGGER.info(
        "Prepared random shared-fractional tasks: tasks=%d skipped=%s patterns=%d prepare=%.3fs",
        len(tasks),
        dict(skipped),
        path_patterns,
        prepare_seconds,
    )

    worker_count = max(1, int(args.workers))
    batch_size = max(1, int(args.batch_size))
    worker_state = {
        "lp": lp,
        "f_values": f_values,
        "g_values": g_values,
        "t_values": t_values,
        "paths_by_word": task_metadata["paths_by_word"],
        "tolerance": args.cut_tolerance,
        "token_only": bool(args.token_only),
    }
    start = time.monotonic()
    checked = 0
    cuts = []
    worker_build = 0.0
    worker_solve = 0.0
    progress_interval = max(1, int(args.progress_interval))

    with cuts_path.open("w", encoding="utf-8") as cuts_file:
        if worker_count == 1:
            init_shared_fractional_worker(worker_state)
            for task in tasks:
                result = shared_fractional_batch_worker([task])[0]
                checked, worker_build, worker_solve = handle_result(
                    result,
                    checked,
                    worker_build,
                    worker_solve,
                    cuts,
                    cuts_file,
                    words,
                    progress_interval,
                    len(tasks),
                    start,
                )
        else:
            context = mp.get_context("fork") if hasattr(os, "fork") else None
            batches = list(chunked(tasks, batch_size))
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=init_shared_fractional_worker,
                initargs=(worker_state,),
            ) as executor:
                futures = [executor.submit(shared_fractional_batch_worker, batch) for batch in batches]
                for future in as_completed(futures):
                    for result in future.result():
                        checked, worker_build, worker_solve = handle_result(
                            result,
                            checked,
                            worker_build,
                            worker_solve,
                            cuts,
                            cuts_file,
                            words,
                            progress_interval,
                            len(tasks),
                            start,
                        )

    elapsed = time.monotonic() - start
    summary = {
        "run_dir": str(run_dir),
        "checkpoint_next_iteration": checkpoint.get("next_iteration"),
        "checkpoint_completed": checkpoint.get("completed"),
        "active_cuts": len(existing_cut_keys),
        "candidate_pairs": len(pair_rows),
        "target_tasks": args.num_pairs,
        "prepared_tasks": len(tasks),
        "checked": checked,
        "cuts": len(cuts),
        "cut_rate": len(cuts) / max(1, checked),
        "prepare_seconds": prepare_seconds,
        "elapsed_seconds": elapsed,
        "pairs_per_second": checked / elapsed if elapsed else 0.0,
        "worker_build_seconds": worker_build,
        "worker_solve_seconds": worker_solve,
        "workers": worker_count,
        "batch_size": batch_size,
        "skipped": dict(skipped),
        "path_patterns": path_patterns,
        "task_metrics": task_metric_summary(task_metadata["rows"]),
        "cuts_path": str(cuts_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log_lines = format_summary(summary)
    text_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    for line in log_lines:
        LOGGER.info(line)
    LOGGER.info("Wrote cuts=%s summary=%s", cuts_path, summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test random short-word pairs using only shared fractional token colors in pair-hull LPs."
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
    parser.add_argument("--pair-min-fractional-shared-colors", type=int, default=2)
    parser.add_argument("--pair-max-pair-rows", type=int, default=250000)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--num-pairs", type=int, default=80000)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress-interval", type=int, default=5000)
    parser.add_argument("--token-only", action="store_true", help="Use only shared fractional token coefficients and deduplicated token-subset rows.")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def load_checkpoint_state(state_dir: Path) -> dict:
    payload = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    payload["existing_cut_keys"] = {
        tupleify_json_key(key)
        for key in payload.get("existing_cut_keys", [])
    }
    return payload


def build_pair_rows(args, lp, f_values, t_values):
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
    return pair_rows


def prepare_random_tasks(args, lp, pair_rows, t_values, existing_cut_keys):
    rng = random.Random(args.random_seed)
    shuffled = list(pair_rows)
    rng.shuffle(shuffled)
    fractional_colors = set(np.flatnonzero((t_values > args.cut_tolerance) & (t_values < 1.0 - args.cut_tolerance)))
    colors_by_word = {}
    paths_by_word = {}
    path_pattern_cache = {}
    tasks = []
    task_rows = []
    skipped = Counter()

    for _, left_word, right_word in shuffled:
        if len(tasks) >= args.num_pairs:
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
        left_paths = paths_by_word[left_word]
        right_paths = paths_by_word[right_word]
        if left_paths is None or right_paths is None:
            skipped["paths"] += 1
            continue
        pair_rows_count = len(left_paths) * len(right_paths)
        if not args.token_only and pair_rows_count > args.pair_max_pair_rows:
            skipped["rows"] += 1
            continue
        if args.token_only:
            tasks.append((left_word, right_word, selected_tokens))
        else:
            tasks.append((left_word, right_word, selected_tokens, None))
        task_rows.append(
            {
                "pair_rows": pair_rows_count,
                "shared_fractional_colors": len(selected_tokens),
                "left_paths": len(left_paths),
                "right_paths": len(right_paths),
            }
        )

    return tasks, {"paths_by_word": paths_by_word, "rows": task_rows}, skipped, len(path_pattern_cache)


def handle_result(
    result,
    checked,
    worker_build,
    worker_solve,
    cuts,
    cuts_file,
    words,
    progress_interval,
    total,
    start,
):
    checked += 1
    worker_build += float(result.get("build_seconds", 0.0))
    worker_solve += float(result.get("solve_seconds", 0.0))
    cut = result.get("cut")
    if cut is not None:
        cuts.append(cut)
        cuts_file.write(json.dumps(cut_record(len(cuts), cut, words), ensure_ascii=False) + "\n")
        cuts_file.flush()
    if checked == 1 or checked % progress_interval == 0 or checked == total:
        elapsed = time.monotonic() - start
        LOGGER.info(
            "shared_fractional progress: checked=%d/%d cuts=%d wall=%.3fs rate=%.2f/s "
            "worker_build=%.3fs worker_solve=%.3fs",
            checked,
            total,
            len(cuts),
            elapsed,
            checked / elapsed if elapsed else 0.0,
            worker_build,
            worker_solve,
        )
    return checked, worker_build, worker_solve


def shared_fractional_worker(task):
    state = SHARED_FRACTIONAL_WORKER_STATE
    if not state["token_only"]:
        return pair_hull_batch_worker([(*task, None) if len(task) == 3 else task])[0]
    left_word, right_word, selected_tokens = task
    left_paths = state["paths_by_word"][left_word]
    right_paths = state["paths_by_word"][right_word]
    start = time.monotonic()
    cut = token_only_pair_cut(
        state["t_values"],
        left_word,
        right_word,
        selected_tokens,
        left_paths,
        right_paths,
        tolerance=state["tolerance"],
    )
    if cut is None:
        return {"cut": None, "build_seconds": time.monotonic() - start, "solve_seconds": 0.0}
    return {"cut": cut, "build_seconds": cut[4], "solve_seconds": cut[5]}


def shared_fractional_batch_worker(tasks):
    return [shared_fractional_worker(task) for task in tasks]


SHARED_FRACTIONAL_WORKER_STATE = {}


def init_shared_fractional_worker(state):
    global SHARED_FRACTIONAL_WORKER_STATE
    SHARED_FRACTIONAL_WORKER_STATE = state
    if not state["token_only"]:
        init_pair_hull_worker(state)


def token_only_pair_cut(t_values, left_word, right_word, selected_tokens, left_paths, right_paths, *, tolerance):
    build_start = time.monotonic()
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    selected_set = set(selected_tokens)
    left_masks = path_token_masks(left_paths, selected_set, selected_position)
    right_masks = path_token_masks(right_paths, selected_set, selected_position)
    required_masks = {left_mask | right_mask for left_mask in left_masks for right_mask in right_masks}

    num_tokens = len(selected_tokens)
    b_pos_offset = 0
    b_neg_offset = num_tokens
    gamma_col = 2 * num_tokens
    num_vars = gamma_col + 1
    a_ub = []
    b_ub = []
    for mask in sorted(required_masks):
        row = [0.0] * num_vars
        for pos in range(num_tokens):
            row[b_pos_offset + pos] = 1.0
            if mask & (1 << pos):
                row[b_neg_offset + pos] = -1.0
        row[gamma_col] = 1.0
        a_ub.append(row)
        b_ub.append(0.0)

    norm = [0.0] * num_vars
    for pos in range(num_tokens):
        norm[b_pos_offset + pos] = 1.0
        norm[b_neg_offset + pos] = 1.0
    a_ub.append(norm)
    b_ub.append(1.0)
    objective = np.zeros(num_vars, dtype=float)
    for pos, token_idx in enumerate(selected_tokens):
        token_value = float(t_values[token_idx])
        objective[b_pos_offset + pos] = -token_value
        objective[b_neg_offset + pos] = token_value
    objective[gamma_col] = -1.0
    bounds = [(0.0, None)] * num_vars
    bounds[gamma_col] = (None, None)
    build_seconds = time.monotonic() - build_start
    solve_start = time.monotonic()
    result = linprog(c=objective, A_ub=np.array(a_ub, dtype=float), b_ub=np.array(b_ub, dtype=float), bounds=bounds, method="highs")
    solve_seconds = time.monotonic() - solve_start
    if not result.success:
        return None
    violation = -float(result.fun)
    if violation <= tolerance:
        return None
    token_coefficients = {}
    for token_idx, pos in selected_position.items():
        coefficient = float(result.x[b_pos_offset + pos] - result.x[b_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            token_coefficients[token_idx] = coefficient
    gamma = float(result.x[gamma_col])
    coefficient_key = (
        round(gamma, 8),
        tuple(round(float(token_coefficients.get(token_idx, 0.0)), 8) for token_idx in selected_tokens),
    )
    entries = [(token_idx, coefficient) for token_idx, coefficient in token_coefficients.items()]
    key = ("shared_fractional_token_only_pair_hull", left_word, right_word, selected_tokens, coefficient_key)
    return (violation, key, entries, -gamma, build_seconds, solve_seconds)


def path_token_masks(paths, selected_set, selected_position):
    masks = set()
    for _, path_tokens in paths:
        mask = 0
        for token_idx in set(path_tokens) & selected_set:
            mask |= 1 << selected_position[token_idx]
        masks.add(mask)
    return masks


def cut_record(rank, cut, words):
    violation, key, entries, rhs = cut
    left_word = key[1]
    right_word = key[2]
    selected_tokens = key[3]
    return {
        "rank": int(rank),
        "violation": float(violation),
        "left_word_idx": int(left_word),
        "right_word_idx": int(right_word),
        "left_word": words[left_word],
        "right_word": words[right_word],
        "num_shared_fractional_colors": len(selected_tokens),
        "rhs": float(rhs),
        "num_entries": len(entries),
    }


def task_metric_summary(rows):
    fields = ("pair_rows", "shared_fractional_colors", "left_paths", "right_paths")
    return {field: quantiles([row[field] for row in rows]) for field in fields}


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


def format_summary(summary):
    return [
        (
            "shared-fractional random pair test: "
            f"checked={summary['checked']} cuts={summary['cuts']} "
            f"cut_rate={summary['cut_rate']:.4f} elapsed={summary['elapsed_seconds']:.3f}s "
            f"rate={summary['pairs_per_second']:.2f}/s"
        ),
        (
            f"prepare={summary['prepare_seconds']:.3f}s candidates={summary['candidate_pairs']} "
            f"prepared={summary['prepared_tasks']} skipped={summary['skipped']}"
        ),
        (
            f"worker_build={summary['worker_build_seconds']:.3f}s "
            f"worker_solve={summary['worker_solve_seconds']:.3f}s "
            f"workers={summary['workers']} batch_size={summary['batch_size']}"
        ),
        f"task_metrics={summary['task_metrics']}",
        f"cuts_path={summary['cuts_path']}",
    ]


if __name__ == "__main__":
    main()
