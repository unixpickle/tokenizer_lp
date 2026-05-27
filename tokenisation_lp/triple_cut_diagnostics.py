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
from pathlib import Path

import numpy as np

from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    all_word_token_colors,
    build_standard_form,
    chunked,
    count_pretokenized_strings,
    enumerate_word_edge_paths_by_pattern,
    pair_hull_current_violation,
    prepare_lp_data,
    rank_short_fractional_words,
    triple_reduced_fractional_edge_hull_cut_from_paths,
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

    existing_records = analyze_existing_triple_cuts(
        lp,
        words,
        tokens,
        f_values,
        g_values,
        t_values,
        checkpoint["existing_cut_keys"],
    )
    existing_path = output_dir / "existing_triple_cuts.jsonl"
    write_jsonl(existing_path, existing_records)
    LOGGER.info("Analyzed %d checkpointed triple cuts", len(existing_records))

    prep_start = time.monotonic()
    tasks, prep_summary, paths_by_word = prepare_scan_tasks(args, lp, f_values, t_values, checkpoint["existing_cut_keys"])
    prep_seconds = time.monotonic() - prep_start
    LOGGER.info("Prepared scan tasks=%d prep=%.1fs summary=%s", len(tasks), prep_seconds, prep_summary)

    scan_records = run_scan(args, lp, words, f_values, g_values, t_values, tasks, paths_by_word)
    scan_path = output_dir / "triple_scan_records.jsonl"
    write_jsonl(scan_path, scan_records)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_next_iteration": checkpoint.get("next_iteration"),
        "active_cuts": len(checkpoint["existing_cut_keys"]),
        "existing_triple_cuts": summarize_existing(existing_records),
        "scan_prep": prep_summary | {"prep_seconds": prep_seconds},
        "scan": summarize_scan(scan_records),
        "existing_records_path": str(existing_path),
        "scan_records_path": str(scan_path),
    }
    summary_path = output_dir / "summary.json"
    log_path = output_dir / "summary.log"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = format_summary(summary)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        LOGGER.info(line)
    LOGGER.info("Wrote summary=%s existing=%s scan=%s", summary_path, existing_path, scan_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze checkpointed and sampled short-word triple hull cuts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8)
    parser.add_argument("--cut-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-words", type=int, default=700)
    parser.add_argument("--max-word-length", type=int, default=12)
    parser.add_argument("--top-words-per-color", type=int, default=48)
    parser.add_argument("--candidate-word-multiplier", type=float, default=16.0)
    parser.add_argument("--candidate-top-words-multiplier", type=float, default=16.0)
    parser.add_argument("--candidate-sample", type=int, default=250000)
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--row-limit", type=int, default=100000)
    parser.add_argument("--token-mode", choices=("shared_all", "at_least_two"), default="at_least_two")
    parser.add_argument("--min-fractional-colors", type=int, default=2)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def load_checkpoint_state(state_dir: Path) -> dict:
    payload = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    payload["existing_cut_keys"] = {tupleify_json_key(key) for key in payload.get("existing_cut_keys", [])}
    return payload


def analyze_existing_triple_cuts(lp, words, tokens, f_values, g_values, t_values, existing_cut_keys):
    records = []
    for key in sorted(existing_cut_keys, key=repr):
        if len(key) < 6 or key[0] != "short_word_triple_hull":
            continue
        word_indices = tuple(int(x) for x in key[1:4])
        selected_tokens = tuple(int(x) for x in key[4])
        coefficient_key = key[5]
        if len(coefficient_key) < 4:
            continue
        token_coefficients = {
            token_idx: float(coefficient)
            for token_idx, coefficient in zip(selected_tokens, coefficient_key[2])
            if abs(float(coefficient)) > 1e-10
        }
        edge_coefficients = {
            int(col_idx): float(coefficient)
            for col_idx, coefficient in coefficient_key[3]
            if abs(float(coefficient)) > 1e-10
        }
        violation_now = pair_hull_current_violation(
            edge_coefficients,
            token_coefficients,
            -float(coefficient_key[1]),
            selected_tokens,
            f_values,
            g_values,
            t_values,
            lp["num_nonfree_edges"],
        )
        records.append(
            {
                "words": word_indices,
                "word_strings": [words[idx] for idx in word_indices],
                "selected_tokens": selected_tokens,
                "selected_token_strings": [tokens[idx].token for idx in selected_tokens],
                "nonzero_token_coefficients": len(token_coefficients),
                "nonzero_edge_coefficients": len(edge_coefficients),
                "token_support_fraction": len(token_coefficients) / max(1, len(selected_tokens)),
                "edge_support": edge_support_summary(lp, edge_coefficients),
                "coefficient_abs": coefficient_abs_summary(edge_coefficients, token_coefficients),
                "current_slack": float(violation_now),
            }
        )
    return records


def edge_support_summary(lp, edge_coefficients):
    num_f = lp["num_nonfree_edges"]
    by_word = Counter()
    by_kind = Counter()
    by_sign = Counter()
    lengths = []
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
        lengths.append(max(1, int(info["end"]) - int(info["start"])))
    return {
        "by_word": dict(by_word),
        "by_kind": dict(by_kind),
        "by_sign": dict(by_sign),
        "distinct_nonfree_edge_tokens": len(token_indices),
        "lengths": quantiles(lengths),
    }


def coefficient_abs_summary(edge_coefficients, token_coefficients):
    edge_abs = [abs(float(value)) for value in edge_coefficients.values()]
    token_abs = [abs(float(value)) for value in token_coefficients.values()]
    return {
        "edge": quantiles(edge_abs),
        "token": quantiles(token_abs),
    }


def prepare_scan_tasks(args, lp, f_values, t_values, existing_cut_keys):
    rng = random.Random(args.random_seed)
    candidate_max_words = max(args.max_words, int(math.ceil(args.max_words * max(1.0, args.candidate_word_multiplier))))
    candidate_top_words = max(
        args.top_words_per_color,
        int(math.ceil(args.top_words_per_color * max(1.0, args.candidate_top_words_multiplier))),
    )
    fractional_colors = set(np.flatnonzero((t_values > args.cut_tolerance) & (t_values < 1.0 - args.cut_tolerance)))
    ranked_words = rank_short_fractional_words(
        lp,
        f_values,
        t_values,
        max_words=candidate_max_words,
        max_word_length=args.max_word_length,
        tolerance=args.cut_tolerance,
    )
    word_colors = {}
    word_color_scores = {}
    color_to_words = defaultdict(list)
    for word_idx in ranked_words:
        scores = defaultdict(float)
        word_weight = float(lp["word_weights"][word_idx])
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            edge_value = float(f_values[edge_idx])
            if not (args.cut_tolerance < edge_value < 1.0 - args.cut_tolerance):
                continue
            info = lp["nonfree_edge_info"][edge_idx]
            token_idx = int(info["token_index"])
            if token_idx not in fractional_colors:
                continue
            scores[token_idx] += min(edge_value, 1.0 - edge_value) * max(1, int(info["end"]) - int(info["start"]))
        word_colors[word_idx] = set(all_word_token_colors(lp, word_idx))
        word_color_scores[word_idx] = dict(scores)
        for token_idx, score in scores.items():
            color_to_words[token_idx].append((word_weight * score, word_idx))
    for rows in color_to_words.values():
        rows.sort(reverse=True)

    triple_candidates = set()
    estimated_candidate_triples = 0
    colors = list(color_to_words)
    rng.shuffle(colors)
    per_color_budget = max(1, int(math.ceil(max(1, args.candidate_sample) / max(1, len(colors)))))
    for token_idx in colors:
        candidate_words = [word_idx for _, word_idx in color_to_words[token_idx][:candidate_top_words]]
        if len(candidate_words) < 3:
            continue
        estimated_candidate_triples += math.comb(len(candidate_words), 3)
        if len(triple_candidates) >= args.candidate_sample:
            continue
        for _ in range(per_color_budget):
            triple_candidates.add(tuple(sorted(rng.sample(candidate_words, 3))))
            if len(triple_candidates) >= args.candidate_sample:
                break

    candidates = list(triple_candidates)
    rng.shuffle(candidates)
    existing_prefixes = {key[:4] for key in existing_cut_keys if len(key) >= 4 and key[0] == "short_word_triple_hull"}
    path_pattern_cache = {}
    paths_by_word = {}
    tasks = []
    skipped = Counter()
    for word_indices in candidates:
        if len(tasks) >= args.sample:
            break
        if ("short_word_triple_hull", *word_indices) in existing_prefixes:
            skipped["existing_cut"] += 1
            continue
        color_sets = [word_colors.setdefault(word_idx, set(all_word_token_colors(lp, word_idx))) for word_idx in word_indices]
        if args.token_mode == "shared_all":
            selected_set = set.intersection(*color_sets) & fractional_colors
        else:
            counts = Counter(token_idx for colors_for_word in color_sets for token_idx in colors_for_word if token_idx in fractional_colors)
            selected_set = {token_idx for token_idx, count in counts.items() if count >= 2}
        selected_tokens = tuple(sorted(selected_set))
        if len(selected_tokens) < args.min_fractional_colors:
            skipped["shared"] += 1
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
            skipped["paths"] += 1
            continue
        tasks.append(
            {
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
                    args.cut_tolerance,
                ),
            }
        )
    return tasks, {
        "candidate_words": candidate_max_words,
        "candidate_top_words": candidate_top_words,
        "candidate_triples_sampled": len(candidates),
        "estimated_candidate_triples": estimated_candidate_triples,
        "sampled_tasks": len(tasks),
        "skipped": dict(skipped),
    }, paths_by_word


def candidate_features(lp, f_values, t_values, word_indices, selected_tokens, word_color_scores, word_colors, paths_by_word, tolerance):
    color_sets = [word_colors[word_idx] for word_idx in word_indices]
    fractional_selected = set(selected_tokens)
    pair_shared = []
    pair_score_products = []
    for i in range(3):
        for j in range(i + 1, 3):
            shared = (color_sets[i] & color_sets[j]) & fractional_selected
            pair_shared.append(len(shared))
            pair_score_products.append(
                sum(
                    min(
                        word_color_scores[word_indices[i]].get(token_idx, 0.0),
                        word_color_scores[word_indices[j]].get(token_idx, 0.0),
                    )
                    * float(t_values[token_idx])
                    for token_idx in shared
                )
            )
    all_shared = len(set.intersection(*color_sets) & fractional_selected)
    edge_vars = 0
    selected_edge_vars = 0
    fractional_edge_signal = 0.0
    for word_idx in word_indices:
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            edge_value = float(f_values[edge_idx])
            if tolerance < edge_value < 1.0 - tolerance:
                edge_vars += 1
                info = lp["nonfree_edge_info"][edge_idx]
                fractional_edge_signal += min(edge_value, 1.0 - edge_value) * max(1, int(info["end"]) - int(info["start"]))
                if int(info["token_index"]) in fractional_selected:
                    selected_edge_vars += 1
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            # Free edge values are not needed for prediction; count is added in the worker.
            pass
    path_counts = [len(paths_by_word[word_idx]) for word_idx in word_indices]
    return {
        "selected_tokens": len(selected_tokens),
        "pair_shared_min": min(pair_shared),
        "pair_shared_max": max(pair_shared),
        "pair_shared_sum": sum(pair_shared),
        "all_shared": all_shared,
        "pair_score_product_min": min(pair_score_products),
        "pair_score_product_max": max(pair_score_products),
        "pair_score_product_sum": sum(pair_score_products),
        "word_weight_min": min(float(lp["word_weights"][word_idx]) for word_idx in word_indices),
        "word_weight_max": max(float(lp["word_weights"][word_idx]) for word_idx in word_indices),
        "word_weight_sum": sum(float(lp["word_weights"][word_idx]) for word_idx in word_indices),
        "word_length_sum": sum(int(lp["word_lengths"][word_idx]) for word_idx in word_indices),
        "word_length_max": max(int(lp["word_lengths"][word_idx]) for word_idx in word_indices),
        "nonfree_fractional_edge_vars": edge_vars,
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


def scan_worker(task):
    state = WORKER_STATE
    word_indices = task["word_indices"]
    selected_tokens = task["selected_tokens"]
    lp = state["lp"]
    f_values = state["f_values"]
    g_values = state["g_values"]
    t_values = state["t_values"]
    num_f = lp["num_nonfree_edges"]
    free_fractional_edges = 0
    for word_idx in word_indices:
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            edge_value = float(g_values[edge_idx])
            if state["tolerance"] < edge_value < 1.0 - state["tolerance"]:
                free_fractional_edges += 1
    start = time.monotonic()
    result = triple_reduced_fractional_edge_hull_cut_from_paths(
        lp,
        f_values,
        g_values,
        t_values,
        word_indices,
        selected_tokens,
        [state["paths_by_word"][word_idx] for word_idx in word_indices],
        tolerance=state["tolerance"],
        max_rows=state["row_limit"],
    )
    wall_seconds = time.monotonic() - start
    row = {
        "words": word_indices,
        "word_strings": [state["words"][word_idx] for word_idx in word_indices],
        "selected_tokens": selected_tokens,
        "features": task["features"] | {"free_fractional_edge_vars": free_fractional_edges},
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
    if "edge_coefficients" not in result:
        return row
    edge_coefficients = result["edge_coefficients"]
    token_coefficients = result["token_coefficients"]
    nonfree_used_tokens = {
        int(lp["nonfree_edge_info"][col_idx]["token_index"])
        for col_idx in edge_coefficients
        if col_idx < num_f
    }
    row.update(
        {
            "has_cut": True,
            "violation": result["violation"],
            "rhs": result["rhs"],
            "nonzero_edge_coefficients": len(edge_coefficients),
            "nonzero_token_coefficients": sum(1 for value in token_coefficients.values() if abs(float(value)) > 1e-10),
            "used_nonfree_edge_token_count": len(nonfree_used_tokens),
            "used_selected_nonfree_edge_token_count": len(nonfree_used_tokens & set(selected_tokens)),
            "edge_support": edge_support_summary(lp, edge_coefficients),
            "coefficient_abs": coefficient_abs_summary(edge_coefficients, token_coefficients),
        }
    )
    return row


def scan_batch_worker(batch):
    return [scan_worker(task) for task in batch]


def run_scan(args, lp, words, f_values, g_values, t_values, tasks, paths_by_word):
    worker_state = {
        "lp": lp,
        "words": words,
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
        futures = [executor.submit(scan_batch_worker, batch) for batch in batches]
        for future in as_completed(futures):
            records.extend(future.result())
            maybe_log_scan_progress(records, len(tasks), start, args.progress_interval)
    return records


def maybe_log_scan_progress(records, total, start, progress_interval):
    checked = len(records)
    if checked == total or (progress_interval > 0 and checked % progress_interval < 32):
        cuts = sum(1 for row in records if row.get("has_cut"))
        skipped = Counter(row.get("skip_reason") for row in records if row.get("skip_reason"))
        LOGGER.info(
            "scan progress: checked=%d/%d cuts=%d skipped=%s elapsed=%.1fs",
            checked,
            total,
            cuts,
            dict(skipped),
            time.monotonic() - start,
        )


def summarize_existing(records):
    return {
        "count": len(records),
        "selected_tokens": quantiles([row["selected_tokens"].__len__() for row in records]),
        "nonzero_token_coefficients": quantiles([row["nonzero_token_coefficients"] for row in records]),
        "nonzero_edge_coefficients": quantiles([row["nonzero_edge_coefficients"] for row in records]),
        "token_support_fraction": quantiles([row["token_support_fraction"] for row in records]),
        "edge_nonfree": quantiles([row["edge_support"]["by_kind"].get("nonfree", 0) for row in records]),
        "edge_free": quantiles([row["edge_support"]["by_kind"].get("free", 0) for row in records]),
        "current_slack": quantiles([row["current_slack"] for row in records]),
        "top_word_triples": sorted(
            (
                {
                    "word_strings": row["word_strings"],
                    "nonzero_edge_coefficients": row["nonzero_edge_coefficients"],
                    "nonzero_token_coefficients": row["nonzero_token_coefficients"],
                    "current_slack": row["current_slack"],
                }
                for row in records
            ),
            key=lambda row: row["current_slack"],
            reverse=True,
        )[:10],
    }


def summarize_scan(records):
    hits = [row for row in records if row.get("has_cut")]
    solved = [row for row in records if row.get("skip_reason") is None]
    skipped = Counter(row.get("skip_reason") for row in records if row.get("skip_reason"))
    feature_names = sorted({key for row in records for key in row.get("features", {})})
    contrasts = {}
    for name in feature_names:
        hit_values = [float(row["features"][name]) for row in hits if name in row["features"]]
        miss_values = [
            float(row["features"][name])
            for row in records
            if not row.get("has_cut") and row.get("skip_reason") is None and name in row["features"]
        ]
        if hit_values or miss_values:
            contrasts[name] = {
                "hit": quantiles(hit_values),
                "noncut_solved": quantiles(miss_values),
            }
    return {
        "checked": len(records),
        "solved_not_row_capped": len(solved),
        "cuts": len(hits),
        "cut_rate": len(hits) / max(1, len(records)),
        "solved_cut_rate": len(hits) / max(1, len(solved)),
        "skipped": dict(skipped),
        "violation": quantiles([row["violation"] for row in hits]),
        "reduced_rows": quantiles([row.get("reduced_rows", 0) for row in records if row.get("reduced_rows") is not None]),
        "edge_vars": quantiles([row.get("edge_vars", 0) for row in records if row.get("edge_vars") is not None]),
        "hit_nonzero_edge_coefficients": quantiles([row["nonzero_edge_coefficients"] for row in hits]),
        "hit_nonzero_token_coefficients": quantiles([row["nonzero_token_coefficients"] for row in hits]),
        "feature_contrasts": contrasts,
        "top_hits": sorted(
            (
                {
                    "word_strings": row["word_strings"],
                    "violation": row["violation"],
                    "reduced_rows": row.get("reduced_rows"),
                    "edge_vars": row.get("edge_vars"),
                    "nonzero_edge_coefficients": row["nonzero_edge_coefficients"],
                    "nonzero_token_coefficients": row["nonzero_token_coefficients"],
                    "features": row["features"],
                }
                for row in hits
            ),
            key=lambda row: row["violation"],
            reverse=True,
        )[:10],
    }


def format_summary(summary):
    existing = summary["existing_triple_cuts"]
    scan = summary["scan"]
    lines = [
        f"checkpoint triples: count={existing['count']} selected_tokens={existing['selected_tokens']} "
        f"nonzero_token_coefficients={existing['nonzero_token_coefficients']} "
        f"nonzero_edge_coefficients={existing['nonzero_edge_coefficients']}",
        f"checkpoint edge support: nonfree={existing['edge_nonfree']} free={existing['edge_free']} "
        f"current_slack={existing['current_slack']}",
        f"scan: checked={scan['checked']} cuts={scan['cuts']} cut_rate={scan['cut_rate']:.5f} "
        f"solved_cut_rate={scan['solved_cut_rate']:.5f} skipped={scan['skipped']}",
        f"scan violation={scan['violation']} rows={scan['reduced_rows']} edge_vars={scan['edge_vars']}",
        f"scan hit supports: edges={scan['hit_nonzero_edge_coefficients']} "
        f"tokens={scan['hit_nonzero_token_coefficients']}",
    ]
    for name in (
        "selected_tokens",
        "pair_shared_min",
        "pair_shared_sum",
        "all_shared",
        "pair_score_product_sum",
        "fractional_edge_signal",
        "selected_nonfree_fractional_edge_vars",
        "path_count_product_log10",
        "word_weight_sum",
    ):
        if name in scan["feature_contrasts"]:
            lines.append(f"feature {name}: {scan['feature_contrasts'][name]}")
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
