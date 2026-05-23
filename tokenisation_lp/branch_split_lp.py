from __future__ import annotations

import argparse
import heapq
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    HighsWarmLpSolver,
    build_standard_form,
    count_pretokenized_strings,
    prepare_lp_data,
    separate_cuts,
)
from tokenisation_lp.pretokenization import (
    DEFAULT_SPECIAL_TOKENS,
    build_pretokenizer,
    byte_level_alphabet,
)


LOGGER = logging.getLogger(__name__)


@dataclass(order=True)
class SearchNode:
    priority: float
    depth: int = field(compare=False)
    fixes: tuple[tuple[int, int], ...] = field(compare=False)
    bound: float = field(default=float("nan"), compare=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore branch-and-bound split bounds for the LP tokenizer relaxation."
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
    parser.add_argument(
        "--cut-rounds",
        type=int,
        default=1,
        help="Cleanup cut rounds to add before branching.",
    )
    parser.add_argument(
        "--cuts-per-round",
        type=int,
        default=500,
        help="Maximum cleanup cuts to add per round.",
    )
    parser.add_argument(
        "--cut-families",
        default="boundary,word_packing",
        help="Comma-separated cleanup cut families to add before branching.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="Number of root fractional token variables to test as one-variable splits.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="Optional branch-and-bound node budget. Use 0 to skip tree search.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth for optional branch-and-bound search.",
    )
    parser.add_argument(
        "--incumbent-tokens",
        type=float,
        default=float("inf"),
        help="Known integral tokenizer token count used for pruning.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
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

    LOGGER.info(
        "Solving branch root: words=%d candidate_tokens=%d nonfree_edges=%d free_edges=%d token_budget=%d",
        len(words),
        lp["num_tokens"],
        lp["num_nonfree_edges"],
        lp["num_free_edges"],
        token_budget,
    )
    cut_families = tuple(family.strip() for family in args.cut_families.split(",") if family.strip())
    root_solution = solve_with_cleanup(
        solver,
        lp,
        cut_rounds=args.cut_rounds,
        cuts_per_round=args.cuts_per_round,
        cut_families=cut_families,
    )
    if not root_solution.success:
        raise RuntimeError(f"Root LP failed: {root_solution.message}")

    root_bound = float(root_solution.fun)
    LOGGER.info("Branch root bound after cleanup: %.3f", root_bound)
    t_offset = lp["num_nonfree_edges"] + lp["num_free_edges"]
    t_values = root_solution.x[t_offset:]
    candidates = fractional_token_candidates(t_values, tokens, max_candidates=args.max_candidates)
    LOGGER.info("Testing %d one-variable split candidates", len(candidates))

    current_fixes: dict[int, int] = {}
    split_results = []
    best_split_bound = root_bound
    start = time.monotonic()
    for rank, token_idx in enumerate(candidates, start=1):
        zero = solve_fixed_node(solver, t_offset, current_fixes, {token_idx: 0})
        one = solve_fixed_node(solver, t_offset, current_fixes, {token_idx: 1})
        split_bound = min(zero, one)
        best_split_bound = max(best_split_bound, split_bound)
        split_results.append((split_bound, zero, one, token_idx))
        LOGGER.info(
            "split %03d token=%r t=%.6f child0=%s child1=%s split_bound=%s best=%s",
            rank,
            tokens[token_idx].token,
            float(t_values[token_idx]),
            format_bound(zero),
            format_bound(one),
            format_bound(split_bound),
            format_bound(best_split_bound),
        )

    restore_fixes(solver, t_offset, current_fixes, {})
    split_results.sort(reverse=True)
    LOGGER.info("Best one-variable split bound: %s", format_bound(best_split_bound))
    for split_bound, zero, one, token_idx in split_results[:10]:
        LOGGER.info(
            "top split token=%r root_t=%.6f child0=%s child1=%s split_bound=%s",
            tokens[token_idx].token,
            float(t_values[token_idx]),
            format_bound(zero),
            format_bound(one),
            format_bound(split_bound),
        )

    if args.max_nodes > 0:
        run_branch_search(
            solver,
            lp,
            tokens,
            t_offset,
            root_solution,
            incumbent=args.incumbent_tokens,
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
        )

    LOGGER.info("Branch split exploration finished in %.3fs", time.monotonic() - start)


def solve_with_cleanup(solver, lp, *, cut_rounds, cuts_per_round, cut_families):
    solution = None
    existing_cut_keys = set()
    for iteration in range(cut_rounds + 1):
        start = time.monotonic()
        solution = solver.solve()
        LOGGER.info(
            "cleanup iteration %d solved in %.3fs objective=%s",
            iteration,
            time.monotonic() - start,
            format_bound(float(solution.fun)) if solution.success else solution.message,
        )
        if not solution.success or iteration == cut_rounds:
            return solution
        cut_matrix, cut_rhs, cut_keys, max_violation = separate_cuts(
            lp,
            solution.x,
            existing_cut_keys=existing_cut_keys,
            max_cuts=cuts_per_round,
            tolerance=1e-6,
            families=cut_families,
        )
        LOGGER.info(
            "cleanup iteration %d: added_cuts=%d max_violation=%.6g",
            iteration,
            len(cut_keys),
            max_violation,
        )
        if not cut_keys:
            return solution
        existing_cut_keys.update(cut_keys)
        solver.add_ub_rows(cut_matrix, cut_rhs)
    return solution


def fractional_token_candidates(t_values, tokens, *, max_candidates: int) -> list[int]:
    rows = []
    for token_idx, value in enumerate(t_values):
        if 1e-6 < value < 1.0 - 1e-6:
            score = min(float(value), 1.0 - float(value)) * tokens[token_idx].token_instance_count
            rows.append((score, tokens[token_idx].token_instance_count, len(tokens[token_idx].token), token_idx))
    rows.sort(reverse=True)
    return [token_idx for _, _, _, token_idx in rows[:max_candidates]]


def solve_fixed_node(solver, t_offset: int, current_fixes: dict[int, int], extra_fixes: dict[int, int]) -> float:
    restore_fixes(solver, t_offset, current_fixes, extra_fixes)
    solution = solver.solve()
    if solution.success:
        return float(solution.fun)
    if "infeasible" in str(solution.message).lower():
        return float("inf")
    LOGGER.warning("Node solve failed: %s", solution.message)
    return float("inf")


def restore_fixes(solver, t_offset: int, current_fixes: dict[int, int], target_fixes: dict[int, int]) -> None:
    for token_idx in sorted(set(current_fixes) - set(target_fixes)):
        solver.change_col_bounds(t_offset + token_idx, 0.0, 1.0)
        del current_fixes[token_idx]
    for token_idx, value in sorted(target_fixes.items()):
        if current_fixes.get(token_idx) == value:
            continue
        solver.change_col_bounds(t_offset + token_idx, float(value), float(value))
        current_fixes[token_idx] = value


def run_branch_search(
    solver,
    lp,
    tokens,
    t_offset: int,
    root_solution,
    *,
    incumbent: float,
    max_nodes: int,
    max_depth: int,
) -> None:
    LOGGER.info(
        "Starting branch-and-bound search: incumbent=%s max_nodes=%d max_depth=%d",
        format_bound(incumbent),
        max_nodes,
        max_depth,
    )
    queue = [SearchNode(priority=float(root_solution.fun), depth=0, fixes=(), bound=float(root_solution.fun))]
    current_fixes: dict[int, int] = {}
    terminal_bounds = []
    processed = 0
    pruned = 0
    while queue and processed < max_nodes:
        node = heapq.heappop(queue)
        processed += 1
        fixed = dict(node.fixes)
        restore_fixes(solver, t_offset, current_fixes, fixed)
        solution = solver.solve()
        if not solution.success:
            pruned += 1
            terminal_bounds.append(float("inf"))
            continue
        bound = float(solution.fun)
        if bound >= incumbent - 1e-6:
            pruned += 1
            terminal_bounds.append(bound)
            LOGGER.info(
                "node %d depth=%d pruned by incumbent bound=%s fixes=%d",
                processed,
                node.depth,
                format_bound(bound),
                len(fixed),
            )
            continue
        if node.depth >= max_depth:
            terminal_bounds.append(bound)
            LOGGER.info(
                "node %d depth=%d frontier bound=%s fixes=%d",
                processed,
                node.depth,
                format_bound(bound),
                len(fixed),
            )
            continue
        token_idx = choose_branch_token(solution.x[t_offset:], tokens, fixed)
        if token_idx is None:
            terminal_bounds.append(bound)
            LOGGER.info("node %d depth=%d integral-token LP bound=%s", processed, node.depth, format_bound(bound))
            continue
        for value in (0, 1):
            child_fixes = tuple(sorted([*fixed.items(), (token_idx, value)]))
            heapq.heappush(
                queue,
                SearchNode(
                    priority=bound,
                    depth=node.depth + 1,
                    fixes=child_fixes,
                    bound=bound,
                ),
            )
        LOGGER.info(
            "node %d depth=%d bound=%s branch_token=%r t=%.6f queue=%d",
            processed,
            node.depth,
            format_bound(bound),
            tokens[token_idx].token,
            float(solution.x[t_offset + token_idx]),
            len(queue),
        )
    restore_fixes(solver, t_offset, current_fixes, {})
    queued_bounds = [node.bound for node in queue]
    certified_bound = min([*terminal_bounds, *queued_bounds], default=float(root_solution.fun))
    LOGGER.info(
        "Branch-and-bound processed=%d pruned=%d remaining=%d certified_bound=%s best_terminal_bound=%s",
        processed,
        pruned,
        len(queue),
        format_bound(certified_bound),
        format_bound(max(terminal_bounds, default=float(root_solution.fun))),
    )


def choose_branch_token(t_values, tokens, fixed: dict[int, int]) -> int | None:
    best = None
    best_score = None
    for token_idx, value in enumerate(t_values):
        if token_idx in fixed or not (1e-6 < value < 1.0 - 1e-6):
            continue
        score = (
            min(float(value), 1.0 - float(value)) * tokens[token_idx].token_instance_count,
            tokens[token_idx].token_instance_count,
            len(tokens[token_idx].token),
        )
        if best_score is None or score > best_score:
            best_score = score
            best = token_idx
    return best


def format_bound(value: float) -> str:
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
