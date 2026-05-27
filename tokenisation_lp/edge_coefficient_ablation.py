from __future__ import annotations

import argparse
import json
import logging
import math
import random
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
    prepare_lp_data,
    short_word_pair_candidates,
    tupleify_json_key,
)
from tokenisation_lp.pretokenization import build_pretokenizer


LOGGER = logging.getLogger(__name__)


VARIANTS = (
    "all_edges",
    "nonfree_only",
    "positive_current",
    "fractional_current",
    "selected_color_nonfree",
    "selected_or_positive",
    "selected_or_fractional",
    "no_edges",
)


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

    tasks, skipped, path_patterns = prepare_tasks(args, lp, f_values, t_values, checkpoint["existing_cut_keys"])
    LOGGER.info("Prepared tasks=%d skipped=%s path_patterns=%d", len(tasks), dict(skipped), path_patterns)

    records_path = output_dir / "edge_ablation_records.jsonl"
    summary_path = output_dir / "edge_ablation_summary.json"
    log_path = output_dir / "edge_ablation.log"

    records = []
    start = time.monotonic()
    with records_path.open("w", encoding="utf-8") as records_file:
        for idx, task in enumerate(tasks, start=1):
            record = analyze_task(args, lp, words, f_values, g_values, t_values, task)
            records.append(record)
            records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if idx == 1 or idx % args.progress_interval == 0 or idx == len(tasks):
                baseline_cuts = sum(1 for row in records if row["variants"]["all_edges"]["has_valid_cut"])
                LOGGER.info(
                    "progress=%d/%d baseline_cuts=%d elapsed=%.1fs",
                    idx,
                    len(tasks),
                    baseline_cuts,
                    time.monotonic() - start,
                )

    summary = summarize(records, elapsed_seconds=time.monotonic() - start)
    summary["run_dir"] = str(run_dir)
    summary["state_dir"] = str(state_dir)
    summary["records_path"] = str(records_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = format_summary(summary)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        LOGGER.info(line)
    LOGGER.info("Wrote records=%s summary=%s log=%s", records_path, summary_path, log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try dropping subsets of short_word_pair_hull edge coefficient variables."
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
    parser.add_argument("--pair-max-pair-rows", type=int, default=10000)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--seed-cuts-jsonl")
    parser.add_argument("--seed-cut-limit", type=int, default=250)
    parser.add_argument("--random-tasks", type=int, default=250)
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--progress-interval", type=int, default=25)
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
    fractional_colors = set(np.flatnonzero((t_values > args.cut_tolerance) & (t_values < 1.0 - args.cut_tolerance)))
    colors_by_word = {}
    paths_by_word = {}
    path_pattern_cache = {}
    tasks = []
    task_keys = set()
    skipped = Counter()

    def add_pair(left_word: int, right_word: int, source: str, score: float | None = None) -> None:
        if left_word > right_word:
            left_word, right_word = right_word, left_word
        key = (left_word, right_word)
        if key in task_keys:
            skipped["duplicate"] += 1
            return
        if any(cut_key[:3] == ("short_word_pair_hull", left_word, right_word) for cut_key in existing_cut_keys):
            skipped["existing"] += 1
            return
        left_colors = colors_by_word.setdefault(left_word, set(all_word_token_colors(lp, left_word)))
        right_colors = colors_by_word.setdefault(right_word, set(all_word_token_colors(lp, right_word)))
        selected_tokens = tuple(sorted((left_colors & right_colors) & fractional_colors))
        if len(selected_tokens) < args.pair_min_fractional_shared_colors:
            skipped["shared"] += 1
            return
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
            return
        pair_rows = len(left_paths) * len(right_paths)
        if pair_rows > args.pair_max_pair_rows:
            skipped["rows"] += 1
            return
        task_keys.add(key)
        tasks.append(
            {
                "left_word": left_word,
                "right_word": right_word,
                "selected_tokens": selected_tokens,
                "left_paths": left_paths,
                "right_paths": right_paths,
                "pair_rows": pair_rows,
                "source": source,
                "score": score,
            }
        )

    if args.seed_cuts_jsonl:
        seed_path = Path(args.seed_cuts_jsonl).expanduser()
        with seed_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len([task for task in tasks if task["source"] == "seed_cut"]) >= args.seed_cut_limit:
                    break
                row = json.loads(line)
                add_pair(int(row["left_word_idx"]), int(row["right_word_idx"]), "seed_cut", float(row.get("violation", 0.0)))

    return_random = max(0, int(args.random_tasks))
    if return_random:
        pair_rows = build_pair_rows(args, lp, f_values, t_values)
        rng = random.Random(args.random_seed)
        shuffled = list(pair_rows)
        rng.shuffle(shuffled)
        random_added = 0
        for score, left_word, right_word in shuffled:
            if random_added >= return_random:
                break
            before = len(tasks)
            add_pair(left_word, right_word, "random", float(score))
            if len(tasks) > before:
                random_added += 1

    return tasks, skipped, len(path_pattern_cache)


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


def analyze_task(args, lp, words, f_values, g_values, t_values, task):
    left_word = task["left_word"]
    right_word = task["right_word"]
    selected_tokens = tuple(task["selected_tokens"])
    left_paths = task["left_paths"]
    right_paths = task["right_paths"]
    variants = {}
    for variant in VARIANTS:
        variants[variant] = solve_variant(
            lp,
            f_values,
            g_values,
            t_values,
            left_word,
            right_word,
            selected_tokens,
            left_paths,
            right_paths,
            variant=variant,
            tolerance=args.cut_tolerance,
        )
    return {
        "source": task["source"],
        "score": task["score"],
        "left_word_idx": left_word,
        "right_word_idx": right_word,
        "left_word": words[left_word],
        "right_word": words[right_word],
        "pair_rows": task["pair_rows"],
        "selected_tokens": selected_tokens,
        "num_selected_tokens": len(selected_tokens),
        "variants": variants,
    }


def solve_variant(
    lp,
    f_values,
    g_values,
    t_values,
    left_word_idx,
    right_word_idx,
    selected_tokens,
    left_paths,
    right_paths,
    *,
    variant: str,
    tolerance: float,
):
    start = time.monotonic()
    num_f = lp["num_nonfree_edges"]
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    selected_set = set(selected_tokens)
    word_columns, current_values, edge_meta = selected_edge_columns(
        lp,
        f_values,
        g_values,
        left_word_idx,
        right_word_idx,
        selected_set,
        variant,
        tolerance,
    )
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
    build_seconds = time.monotonic() - start
    solve_start = time.monotonic()
    result = linprog(c=objective, A_ub=constraints, b_ub=np.array(rhs, dtype=float), bounds=bounds, method="highs")
    solve_seconds = time.monotonic() - solve_start
    wall_seconds = time.monotonic() - start
    base = {
        "edge_vars": num_edge_vars,
        "token_vars": num_token_vars,
        "vars": num_vars,
        "dedup_rows": row_idx,
        "original_rows": len(left_paths) * len(right_paths),
        "edge_meta": edge_meta,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "wall_seconds": wall_seconds,
        "success": bool(result.success),
        "has_cut": False,
        "has_valid_cut": False,
    }
    if not result.success:
        base["message"] = result.message
        return base

    reduced_violation = -float(result.fun)
    base["reduced_violation"] = reduced_violation
    if reduced_violation <= tolerance:
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
    base.update(
        {
            "has_cut": True,
            "has_valid_cut": bool(max_slack <= 1e-7 and current_violation > tolerance),
            "current_violation": current_violation,
            "max_full_row_slack": max_slack,
            "rhs": rhs_value,
            "nonzero_edge_coefficients": len(edge_coefficients),
            "nonzero_token_coefficients": len(token_coefficients),
            "support": support_summary(lp, edge_coefficients, selected_set, f_values, g_values),
        }
    )
    return base


def selected_edge_columns(lp, f_values, g_values, left_word_idx, right_word_idx, selected_set, variant, tolerance):
    num_f = lp["num_nonfree_edges"]
    columns = []
    values = []
    meta = Counter()

    def maybe_add(col_idx, value, is_nonfree, token_idx=None):
        keep = False
        if variant == "all_edges":
            keep = True
        elif variant == "nonfree_only":
            keep = is_nonfree
        elif variant == "positive_current":
            keep = value > tolerance
        elif variant == "fractional_current":
            keep = tolerance < value < 1.0 - tolerance
        elif variant == "selected_color_nonfree":
            keep = is_nonfree and token_idx in selected_set
        elif variant == "selected_or_positive":
            keep = (is_nonfree and token_idx in selected_set) or value > tolerance
        elif variant == "selected_or_fractional":
            keep = (is_nonfree and token_idx in selected_set) or (tolerance < value < 1.0 - tolerance)
        elif variant == "no_edges":
            keep = False
        else:
            raise ValueError(f"unknown variant {variant!r}")
        if not keep:
            return
        columns.append(col_idx)
        values.append(float(value))
        meta["nonfree" if is_nonfree else "free"] += 1
        if value > tolerance:
            meta["positive"] += 1
        if tolerance < value < 1.0 - tolerance:
            meta["fractional"] += 1
        if is_nonfree and token_idx in selected_set:
            meta["selected_color"] += 1

    for word_idx in (left_word_idx, right_word_idx):
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            token_idx = lp["nonfree_edge_info"][edge_idx]["token_index"]
            maybe_add(edge_idx, float(f_values[edge_idx]), True, token_idx)
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            maybe_add(num_f + edge_idx, float(g_values[edge_idx]), False, None)
    return columns, values, dict(meta)


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


def support_summary(lp, edge_coefficients, selected_set, f_values, g_values):
    num_f = lp["num_nonfree_edges"]
    counts = Counter()
    for col_idx in edge_coefficients:
        if col_idx < num_f:
            counts["nonfree"] += 1
            value = float(f_values[col_idx])
            if lp["nonfree_edge_info"][col_idx]["token_index"] in selected_set:
                counts["selected_color"] += 1
        else:
            counts["free"] += 1
            value = float(g_values[col_idx - num_f])
        if value > 1e-6:
            counts["positive"] += 1
        if 1e-6 < value < 1.0 - 1e-6:
            counts["fractional"] += 1
    return dict(counts)


def summarize(records, *, elapsed_seconds):
    baseline_cut_indices = {
        idx for idx, row in enumerate(records) if row["variants"]["all_edges"].get("has_valid_cut")
    }
    variant_summary = {}
    for variant in VARIANTS:
        rows = [row["variants"][variant] for row in records]
        valid_indices = {idx for idx, row in enumerate(records) if row["variants"][variant].get("has_valid_cut")}
        variant_summary[variant] = {
            "success": sum(1 for row in rows if row.get("success")),
            "cuts": sum(1 for row in rows if row.get("has_cut")),
            "valid_cuts": len(valid_indices),
            "invalid_cuts": sum(1 for row in rows if row.get("has_cut") and not row.get("has_valid_cut")),
            "baseline_cut_coverage": len(valid_indices & baseline_cut_indices),
            "baseline_cut_coverage_rate": len(valid_indices & baseline_cut_indices) / max(1, len(baseline_cut_indices)),
            "median_edge_vars": quantile([row.get("edge_vars", 0) for row in rows], 0.5),
            "median_dedup_rows": quantile([row.get("dedup_rows", 0) for row in rows], 0.5),
            "median_wall_seconds": quantile([row.get("wall_seconds", 0.0) for row in rows], 0.5),
            "total_wall_seconds": float(sum(row.get("wall_seconds", 0.0) for row in rows)),
        }
    by_source = defaultdict(int)
    for row in records:
        by_source[row["source"]] += 1
    return {
        "tasks": len(records),
        "tasks_by_source": dict(by_source),
        "baseline_valid_cuts": len(baseline_cut_indices),
        "elapsed_seconds": elapsed_seconds,
        "variants": variant_summary,
    }


def quantile(values, q):
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def format_summary(summary):
    lines = [
        (
            "edge ablation: "
            f"tasks={summary['tasks']} baseline_valid_cuts={summary['baseline_valid_cuts']} "
            f"elapsed={summary['elapsed_seconds']:.3f}s"
        )
    ]
    for variant, row in summary["variants"].items():
        lines.append(
            f"{variant}: valid_cuts={row['valid_cuts']} coverage={row['baseline_cut_coverage']}/"
            f"{summary['baseline_valid_cuts']} ({row['baseline_cut_coverage_rate']:.3f}) "
            f"median_edge_vars={row['median_edge_vars']:.1f} median_rows={row['median_dedup_rows']:.1f} "
            f"median_wall={row['median_wall_seconds']:.4f}s invalid={row['invalid_cuts']}"
        )
    lines.append(f"records_path={summary['records_path']}")
    return lines


if __name__ == "__main__":
    main()
