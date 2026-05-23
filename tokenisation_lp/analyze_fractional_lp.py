from __future__ import annotations

import argparse
import logging
from collections import defaultdict

import numpy as np
from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    HighsWarmLpSolver,
    build_standard_form,
    count_pretokenized_strings,
    separate_cuts,
    prepare_lp_data,
)
from tokenisation_lp.pretokenization import (
    DEFAULT_SPECIAL_TOKENS,
    build_pretokenizer,
    byte_level_alphabet,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the tokenizer LP and inspect fractional variables and violated valid cuts."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument(
        "--pretokenizer",
        default="nanochat",
        choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"),
    )
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--cut-rounds", type=int, default=0)
    parser.add_argument("--cuts-per-round", type=int, default=500)
    parser.add_argument(
        "--cut-families",
        default="boundary,word_packing",
        help="Comma-separated families to add before final diagnostics.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--lp-solution-cache-dir",
        default=None,
        help="Optional directory for caching identical LP solutions.",
    )
    parser.add_argument(
        "--lp-solver",
        choices=("highspy",),
        default="highspy",
        help="Diagnostic solver backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    texts = load_texts(args.data_dir)
    pretokenizer, _ = build_pretokenizer(args.pretokenizer)
    word_counts = count_pretokenized_strings(texts, pretokenizer)
    words = list(word_counts)
    freqs = [word_counts[word] for word in words]
    max_token_length = None if args.max_token_length == 0 else args.max_token_length

    edges, free_edges, num_vertices, tokens = prepare_lp_data(
        words,
        freqs,
        min_token_count=args.min_token_count,
        max_token_length=max_token_length,
    )
    lp = build_standard_form(edges, freqs, tokens, free_edges, num_vertices)
    token_budget = args.vocab_size - len(DEFAULT_SPECIAL_TOKENS) - len(byte_level_alphabet())
    if token_budget < 0:
        raise ValueError("vocab size is too small for specials plus byte-level alphabet")

    b_ub = lp["b_ub"].copy()
    b_ub[lp["budget_row"]] = float(token_budget)
    cut_families = tuple(
        family.strip()
        for family in args.cut_families.split(",")
        if family.strip()
    )

    LOGGER.info(
        "Solving LP: words=%d candidate_tokens=%d nonfree_edges=%d free_edges=%d token_budget=%d",
        len(words),
        lp["num_tokens"],
        lp["num_nonfree_edges"],
        lp["num_free_edges"],
        token_budget,
    )
    a_ub = lp["A_ub"]
    solver = HighsWarmLpSolver(
        c=lp["c"],
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=lp["A_eq"],
        b_eq=lp["b_eq"],
        lb=lp["lb"],
        ub=lp["ub"],
        cache_dir=args.lp_solution_cache_dir,
    )
    existing_cut_keys = set()
    solution = None
    for iteration in range(args.cut_rounds + 1):
        solution = solver.solve()
        if not solution.success:
            raise RuntimeError(f"LP solve failed: {solution.message}")

        if iteration == args.cut_rounds:
            break

        cut_matrix, cut_rhs, cut_keys, max_violation = separate_cuts(
            lp,
            solution.x,
            existing_cut_keys=existing_cut_keys,
            max_cuts=args.cuts_per_round,
            tolerance=args.eps,
            families=cut_families,
        )
        LOGGER.info(
            "Diagnostic cut round %d: objective=%.3f added_cuts=%d max_violation=%.6g",
            iteration,
            float(solution.fun),
            len(cut_keys),
            max_violation,
        )
        if not cut_keys:
            break
        existing_cut_keys.update(cut_keys)
        solver.add_ub_rows(cut_matrix, cut_rhs)

    if solution is None:
        raise RuntimeError("LP was not solved")
    if not solution.success:
        raise RuntimeError(f"LP solve failed: {solution.message}")

    LOGGER.info("Final diagnostic LP includes %d added cuts", len(existing_cut_keys))
    analyze_solution(args, lp, solution.x, tokens, words)


def analyze_solution(args, lp, x_values, tokens, words) -> None:
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    f_values = x_values[:num_f]
    g_values = x_values[num_f : num_f + num_g]
    t_values = x_values[num_f + num_g :]

    LOGGER.info("LP objective lower bound: %.3f tokens", float(lp["c"] @ x_values))
    log_fractionality("t", t_values, args.eps)
    log_fractionality("f", f_values, args.eps)
    log_fractionality("g", g_values, args.eps)
    log_fractional_tokens(t_values, tokens, args.top, args.eps)
    log_fractional_edges(f_values, t_values, lp, tokens, words, args.top, args.eps)

    boundary_violations = same_color_boundary_violations(lp, f_values, t_values, args.eps)
    log_boundary_violations(boundary_violations, f_values, lp, tokens, words, args.top)

    packing_violations = same_color_word_packing_violations(lp, f_values, t_values, args.eps)
    log_packing_violations(packing_violations, lp, tokens, words, args.top)

    aggregate_violations = aggregate_boundary_activation_violations(lp, f_values, t_values, args.eps)
    log_aggregate_boundary_violations(aggregate_violations, tokens, words, args.top)


def log_fractionality(name: str, values: np.ndarray, eps: float) -> None:
    fractional = (values > eps) & (values < 1.0 - eps)
    positive = values > eps
    LOGGER.info(
        "%s variables: total=%d positive=%d fractional=%d sum=%.3f",
        name,
        len(values),
        int(positive.sum()),
        int(fractional.sum()),
        float(values.sum()),
    )


def log_fractional_tokens(t_values, tokens, top: int, eps: float) -> None:
    rows = [
        (float(value), tokens[idx].token, tokens[idx].token_instance_count)
        for idx, value in enumerate(t_values)
        if eps < value < 1.0 - eps
    ]
    rows.sort(key=lambda row: min(row[0], 1.0 - row[0]), reverse=True)
    LOGGER.info("Top fractional token-selection variables:")
    for value, token, count in rows[:top]:
        LOGGER.info("  t=%.6f count=%d token=%r", value, count, token)


def log_fractional_edges(f_values, t_values, lp, tokens, words, top: int, eps: float) -> None:
    rows = []
    for edge_idx, value in enumerate(f_values):
        if eps < value < 1.0 - eps:
            info = lp["nonfree_edge_info"][edge_idx]
            rows.append((float(value), edge_idx, info))
    rows.sort(key=lambda row: min(row[0], 1.0 - row[0]), reverse=True)
    LOGGER.info("Top fractional non-free edge variables:")
    for value, edge_idx, info in rows[:top]:
        token = tokens[info["token_index"]]
        word = words[info["word_idx"]]
        LOGGER.info(
            "  f=%.6f edge=%d word=%r span=(%d,%d) token=%r t=%.6f",
            value,
            edge_idx,
            word,
            info["start"],
            info["end"],
            info["token"],
            float(t_values[info["token_index"]]),
        )


def same_color_boundary_violations(lp, f_values, t_values, eps: float):
    violations = []
    for key, edge_indices in lp["boundary_crossings"].items():
        token_idx = key[2]
        lhs = float(f_values[edge_indices].sum())
        rhs = float(t_values[token_idx])
        violation = lhs - rhs
        if violation > eps:
            violations.append((violation, lhs, rhs, key, edge_indices))
    violations.sort(key=lambda row: row[0], reverse=True)
    return violations


def log_boundary_violations(violations, f_values, lp, tokens, words, top: int) -> None:
    LOGGER.info("Same-colour byte-boundary cuts violated: %d", len(violations))
    for violation, lhs, rhs, (word_idx, boundary, token_idx), edge_indices in violations[:top]:
        LOGGER.info(
            "  violation=%.6f lhs=%.6f rhs=%.6f word=%r boundary=%d token=%r crossings=%d",
            violation,
            lhs,
            rhs,
            words[word_idx],
            boundary,
            tokens[token_idx].token,
            len(edge_indices),
        )
        for edge_idx in edge_indices[:5]:
            info = lp["nonfree_edge_info"][edge_idx]
            LOGGER.info(
                "    edge=%d f=%.6f span=(%d,%d)",
                edge_idx,
                float(f_values[edge_idx]),
                info["start"],
                info["end"],
            )


def same_color_word_packing_violations(lp, f_values, t_values, eps: float):
    grouped = defaultdict(list)
    for edge_idx, info in enumerate(lp["nonfree_edge_info"]):
        grouped[(info["word_idx"], info["token_index"])].append(edge_idx)

    violations = []
    for (word_idx, token_idx), edge_indices in grouped.items():
        max_pack = max_non_overlapping_intervals(
            (lp["nonfree_edge_info"][edge_idx]["start"], lp["nonfree_edge_info"][edge_idx]["end"])
            for edge_idx in edge_indices
        )
        if max_pack <= 1:
            continue
        lhs = float(f_values[edge_indices].sum())
        rhs = float(max_pack * t_values[token_idx])
        violation = lhs - rhs
        if violation > eps:
            violations.append((violation, lhs, rhs, max_pack, word_idx, token_idx, edge_indices))
    violations.sort(key=lambda row: row[0], reverse=True)
    return violations


def log_packing_violations(violations, lp, tokens, words, top: int) -> None:
    LOGGER.info("Same-colour word-packing cuts violated: %d", len(violations))
    for violation, lhs, rhs, max_pack, word_idx, token_idx, edge_indices in violations[:top]:
        LOGGER.info(
            "  violation=%.6f lhs=%.6f rhs=%.6f max_pack=%d word=%r token=%r occurrences=%d",
            violation,
            lhs,
            rhs,
            max_pack,
            words[word_idx],
            tokens[token_idx].token,
            len(edge_indices),
        )


def aggregate_boundary_activation_violations(lp, f_values, t_values, eps: float):
    by_word_boundary = defaultdict(lambda: defaultdict(float))
    for (word_idx, boundary, token_idx), edge_indices in lp["boundary_crossings"].items():
        flow = float(f_values[edge_indices].sum())
        if flow > eps:
            by_word_boundary[(word_idx, boundary)][token_idx] += flow

    violations = []
    for (word_idx, boundary), token_flow in by_word_boundary.items():
        positive_terms = [
            (token_idx, flow, float(t_values[token_idx]), flow - float(t_values[token_idx]))
            for token_idx, flow in token_flow.items()
            if flow - float(t_values[token_idx]) > eps
        ]
        if not positive_terms:
            continue
        lhs = sum(term[1] for term in positive_terms)
        rhs = sum(term[2] for term in positive_terms)
        violations.append((lhs - rhs, lhs, rhs, word_idx, boundary, positive_terms))
    violations.sort(key=lambda row: row[0], reverse=True)
    return violations


def log_aggregate_boundary_violations(violations, tokens, words, top: int) -> None:
    LOGGER.info("Aggregate boundary activation cuts violated: %d", len(violations))
    for violation, lhs, rhs, word_idx, boundary, terms in violations[:top]:
        LOGGER.info(
            "  violation=%.6f lhs=%.6f rhs=%.6f word=%r boundary=%d colors=%d",
            violation,
            lhs,
            rhs,
            words[word_idx],
            boundary,
            len(terms),
        )
        for token_idx, flow, token_value, excess in sorted(terms, key=lambda row: row[3], reverse=True)[:5]:
            LOGGER.info(
                "    token=%r flow=%.6f t=%.6f excess=%.6f",
                tokens[token_idx].token,
                flow,
                token_value,
                excess,
            )


def max_non_overlapping_intervals(intervals) -> int:
    count = 0
    end_so_far = -1
    for start, end in sorted(intervals, key=lambda interval: (interval[1], interval[0])):
        if start >= end_so_far:
            count += 1
            end_so_far = end
    return count


if __name__ == "__main__":
    main()
