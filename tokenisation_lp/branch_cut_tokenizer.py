from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from tokenisation_lp.branch_split_lp import choose_branch_token, format_bound, restore_fixes
from tokenisation_lp.corpus import load_texts
from tokenisation_lp.dp_tokenizer import LpDpTokenizer
from tokenisation_lp.evaluation import CompressionStats, evaluate_texts
from tokenisation_lp.lp_training import (
    CutSeparationConfig,
    HighsWarmLpSolver,
    build_standard_form,
    candidates_from_solution,
    count_pretokenized_strings,
    prepare_lp_data,
    round_lp_tokens,
    separate_cuts,
)
from tokenisation_lp.pretokenization import (
    DEFAULT_SPECIAL_TOKENS,
    DEFAULT_UNK_TOKEN,
    build_pretokenizer,
    byte_level_alphabet,
)


LOGGER = logging.getLogger(__name__)


@dataclass(order=True)
class BranchNode:
    priority: float
    depth: int = field(compare=False)
    fixes: tuple[tuple[int, int], ...] = field(compare=False)
    parent_bound: float = field(default=float("nan"), compare=False)


@dataclass
class Incumbent:
    stats: CompressionStats | None = None
    tokenizer: LpDpTokenizer | None = None
    path: Path | None = None
    node_id: int | None = None
    bound: float = float("inf")

    @property
    def tokens(self) -> float:
        if self.stats is None:
            return float("inf")
        return float(self.stats.tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a discrete LP-DP tokenizer with branch-and-cut-guided rounding. "
            "The search solves LP nodes with globally valid cuts, rounds each node "
            "under its branch fixes, evaluates the tokenizer, and saves the best."
        )
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing training text files.")
    parser.add_argument(
        "--eval-dir",
        help="Optional directory containing evaluation text files. Defaults to --data-dir.",
    )
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument(
        "--output-dir",
        default="branch_cut_tokenizer_runs",
        help="Directory where branch-and-cut tokenizer artifacts are written.",
    )
    parser.add_argument(
        "--pretokenizer",
        default="nanochat",
        choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"),
    )
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=8,
        help="Maximum LP candidate token length. Use 0 to consider every substring.",
    )
    parser.add_argument("--max-nodes", type=int, default=32, help="Branch node budget.")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum branch depth.")
    parser.add_argument(
        "--cut-rounds",
        type=int,
        default=1,
        help="Global cut rounds added at the root before branching.",
    )
    parser.add_argument(
        "--node-cut-rounds",
        type=int,
        default=0,
        help=(
            "Extra global cut rounds separated from branch node solutions. Cuts are valid "
            "globally, so they remain active for later nodes."
        ),
    )
    parser.add_argument("--cuts-per-round", type=int, default=500)
    parser.add_argument("--node-cuts-per-round", type=int, default=100)
    parser.add_argument("--cut-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--cut-families",
        default="boundary,word_packing",
        help=(
            "Comma-separated cut families. Useful expensive choices include "
            "short_word_full_hull and short_word_pair_hull."
        ),
    )
    parser.add_argument(
        "--incumbent-tokenizer",
        help="Optional existing LP-DP tokenizer JSON used to seed best-so-far.",
    )
    parser.add_argument(
        "--incumbent-tokens",
        type=float,
        default=None,
        help=(
            "Optional known training-token incumbent for bound pruning. If omitted, "
            "the best evaluated tokenizer token count is used."
        ),
    )
    parser.add_argument(
        "--disable-bound-pruning",
        action="store_true",
        help="Evaluate and branch nodes even when the LP lower bound is worse than the incumbent.",
    )
    parser.add_argument(
        "--save-node-tokenizers",
        action="store_true",
        help="Save every rounded node tokenizer, not just the best one.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="Worker processes for DP tokenizer evaluation.",
    )
    parser.add_argument(
        "--highs-threads",
        type=int,
        default=0,
        help="HiGHS thread count. 0 lets HiGHS choose.",
    )
    parser.add_argument(
        "--highs-parallel",
        default="on",
        choices=("on", "off", "choose"),
        help="HiGHS parallel option for the live branch-and-cut solver.",
    )
    parser.add_argument(
        "--lp-solution-cache-dir",
        default=None,
        help="Optional exact LP cache directory. Normally leave unset for live warm starts.",
    )
    parser.add_argument("--short-word-full-hull-max-words", type=int, default=250)
    parser.add_argument("--short-word-full-hull-max-length", type=int, default=8)
    parser.add_argument("--short-word-full-hull-max-colors", type=int, default=64)
    parser.add_argument("--short-word-pair-hull-max-words", type=int, default=500)
    parser.add_argument("--short-word-pair-hull-max-length", type=int, default=12)
    parser.add_argument("--short-word-pair-hull-max-colors", type=int, default=96)
    parser.add_argument("--short-word-pair-hull-max-pair-rows", type=int, default=250000)
    parser.add_argument("--short-word-pair-hull-max-pairs", type=int, default=800)
    parser.add_argument("--short-word-pair-hull-top-words-per-color", type=int, default=36)
    parser.add_argument("--short-word-pair-hull-workers", type=int, default=0)
    parser.add_argument("--short-word-pair-hull-batch-size", type=int, default=32)
    parser.add_argument("--short-word-pair-hull-min-fractional-shared-colors", type=int, default=1)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_texts = load_texts(args.data_dir)
    eval_texts = load_texts(args.eval_dir or args.data_dir)
    max_token_length = None if args.max_token_length == 0 else args.max_token_length
    cut_families = tuple(family.strip() for family in args.cut_families.split(",") if family.strip())
    cut_config = build_cut_config(args)

    pretokenizer, _ = build_pretokenizer(args.pretokenizer)
    word_counts = count_pretokenized_strings(train_texts, pretokenizer)
    words = list(word_counts)
    freqs = [word_counts[word] for word in words]
    edges, free_edges, num_vertices, tokens = prepare_lp_data(
        words,
        freqs,
        min_token_count=args.min_token_count,
        max_token_length=max_token_length,
    )
    lp = build_standard_form(edges, freqs, tokens, free_edges, num_vertices)
    base_vocab = base_vocabulary()
    token_budget = args.vocab_size - len(base_vocab)
    if token_budget < 0:
        raise ValueError("vocab size is too small for specials plus byte-level alphabet")
    t_offset = lp["num_nonfree_edges"] + lp["num_free_edges"]

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
        cache_dir=args.lp_solution_cache_dir,
        highs_threads=args.highs_threads,
        highs_parallel=args.highs_parallel,
    )

    LOGGER.info(
        "Branch-and-cut tokenizer: train_files=%d eval_files=%d words=%d candidate_tokens=%d "
        "nonfree_edges=%d free_edges=%d token_budget=%d",
        len(train_texts),
        len(eval_texts),
        len(words),
        lp["num_tokens"],
        lp["num_nonfree_edges"],
        lp["num_free_edges"],
        token_budget,
    )

    incumbent = load_incumbent(args, output_dir, eval_texts)
    existing_cut_keys: set[tuple] = set()
    current_fixes: dict[int, int] = {}
    start_time = time.monotonic()

    root_solution, root_added_cuts = solve_with_global_cuts(
        solver,
        lp,
        existing_cut_keys=existing_cut_keys,
        rounds=args.cut_rounds,
        cuts_per_round=args.cuts_per_round,
        tolerance=args.cut_tolerance,
        families=cut_families,
        config=cut_config,
        label="root",
    )
    if not root_solution.success:
        raise RuntimeError(f"Root LP failed: {root_solution.message}")
    root_bound = float(root_solution.fun)
    LOGGER.info("Root bound after cuts: %s root_added_cuts=%d", format_bound(root_bound), root_added_cuts)

    queue = [BranchNode(priority=root_bound, depth=0, fixes=(), parent_bound=root_bound)]
    node_metadata = []
    processed = 0
    pruned = 0
    infeasible = 0
    terminal_bounds = []

    while queue and processed < args.max_nodes:
        node = heapq.heappop(queue)
        processed += 1
        fixed = dict(node.fixes)
        restore_fixes(solver, t_offset, current_fixes, fixed)

        solution, added_cuts = solve_with_global_cuts(
            solver,
            lp,
            existing_cut_keys=existing_cut_keys,
            rounds=args.node_cut_rounds,
            cuts_per_round=args.node_cuts_per_round,
            tolerance=args.cut_tolerance,
            families=cut_families,
            config=cut_config,
            label=f"node {processed}",
        )
        if not solution.success:
            infeasible += 1
            terminal_bounds.append(float("inf"))
            LOGGER.info("node %d depth=%d infeasible_or_failed=%s", processed, node.depth, solution.message)
            continue

        bound = float(solution.fun)
        fractional_colors = count_fractional_colors(lp, solution.x)
        prune_by_bound = (
            not args.disable_bound_pruning
            and math.isfinite(incumbent.tokens)
            and bound >= incumbent.tokens - 1e-6
        )
        rounded_stats = None
        if not prune_by_bound:
            rounded_tokenizer = round_node_solution(
                tokens=tokens,
                lp=lp,
                solution_x=solution.x,
                fixed=fixed,
                base_vocab=base_vocab,
                token_budget=token_budget,
                pretokenizer_mode=args.pretokenizer,
            )
            if rounded_tokenizer is not None:
                rounded_stats = evaluate_node_tokenizer(
                    processed,
                    rounded_tokenizer,
                    eval_texts,
                    eval_workers=args.eval_workers,
                    output_dir=output_dir,
                    save_node_tokenizers=args.save_node_tokenizers,
                )
                if rounded_stats.tokens < incumbent.tokens:
                    save_best_incumbent(
                        incumbent,
                        tokenizer=rounded_tokenizer,
                        stats=rounded_stats,
                        output_dir=output_dir,
                        node_id=processed,
                        bound=bound,
                        fixed=fixed,
                    )
        else:
            pruned += 1

        branch_token_idx = None
        if prune_by_bound:
            terminal_bounds.append(bound)
            LOGGER.info(
                "node %d depth=%d pruned_by_bound bound=%s incumbent_tokens=%s fixes=%d "
                "fractional_colors=%d added_cuts=%d",
                processed,
                node.depth,
                format_bound(bound),
                format_bound(incumbent.tokens),
                len(fixed),
                fractional_colors,
                added_cuts,
            )
        elif node.depth >= args.max_depth:
            terminal_bounds.append(bound)
        else:
            branch_token_idx = choose_branch_token(solution.x[t_offset:], tokens, fixed)
            if branch_token_idx is None:
                terminal_bounds.append(bound)
            else:
                for value in (0, 1):
                    child_fixes = tuple(sorted([*fixed.items(), (branch_token_idx, value)]))
                    heapq.heappush(
                        queue,
                        BranchNode(
                            priority=bound,
                            depth=node.depth + 1,
                            fixes=child_fixes,
                            parent_bound=bound,
                        ),
                    )

        rounded_tokens = rounded_stats.tokens if rounded_stats is not None else None
        LOGGER.info(
            "node %d depth=%d bound=%s rounded_tokens=%s best_tokens=%s fixes=%d "
            "fractional_colors=%d added_cuts=%d branch_token=%s branch_value=%.6f queue=%d",
            processed,
            node.depth,
            format_bound(bound),
            rounded_tokens if rounded_tokens is not None else "-",
            format_bound(incumbent.tokens),
            len(fixed),
            fractional_colors,
            added_cuts,
            repr(tokens[branch_token_idx].token) if branch_token_idx is not None else "-",
            float(solution.x[t_offset + branch_token_idx]) if branch_token_idx is not None else float("nan"),
            len(queue),
        )
        node_metadata.append(
            {
                "node": processed,
                "depth": node.depth,
                "bound": bound,
                "rounded_tokens": rounded_tokens,
                "best_tokens": None if not math.isfinite(incumbent.tokens) else incumbent.tokens,
                "fixes": [[int(k), int(v)] for k, v in fixed.items()],
                "fractional_colors": fractional_colors,
                "added_cuts": added_cuts,
                "branch_token": tokens[branch_token_idx].token if branch_token_idx is not None else None,
                "branch_value": (
                    float(solution.x[t_offset + branch_token_idx]) if branch_token_idx is not None else None
                ),
                "queue": len(queue),
            }
        )
        write_search_metadata(
            output_dir,
            args=args,
            processed=processed,
            pruned=pruned,
            infeasible=infeasible,
            queue=queue,
            terminal_bounds=terminal_bounds,
            root_bound=root_bound,
            incumbent=incumbent,
            node_metadata=node_metadata,
            elapsed=time.monotonic() - start_time,
        )

    restore_fixes(solver, t_offset, current_fixes, {})
    write_search_metadata(
        output_dir,
        args=args,
        processed=processed,
        pruned=pruned,
        infeasible=infeasible,
        queue=queue,
        terminal_bounds=terminal_bounds,
        root_bound=root_bound,
        incumbent=incumbent,
        node_metadata=node_metadata,
        elapsed=time.monotonic() - start_time,
    )
    certified_bound = min([*terminal_bounds, *(node.parent_bound for node in queue)], default=root_bound)
    LOGGER.info(
        "Branch-and-cut tokenizer finished: processed=%d pruned=%d infeasible=%d remaining=%d "
        "certified_bound=%s best_tokens=%s best_path=%s elapsed=%.3fs",
        processed,
        pruned,
        infeasible,
        len(queue),
        format_bound(certified_bound),
        format_bound(incumbent.tokens),
        incumbent.path,
        time.monotonic() - start_time,
    )


def build_cut_config(args: argparse.Namespace) -> CutSeparationConfig:
    return CutSeparationConfig(
        short_word_full_hull_max_words=args.short_word_full_hull_max_words,
        short_word_full_hull_max_length=args.short_word_full_hull_max_length,
        short_word_full_hull_max_colors=args.short_word_full_hull_max_colors,
        short_word_pair_hull_max_words=args.short_word_pair_hull_max_words,
        short_word_pair_hull_max_length=args.short_word_pair_hull_max_length,
        short_word_pair_hull_max_colors=args.short_word_pair_hull_max_colors,
        short_word_pair_hull_max_pair_rows=args.short_word_pair_hull_max_pair_rows,
        short_word_pair_hull_max_pairs=args.short_word_pair_hull_max_pairs,
        short_word_pair_hull_top_words_per_color=args.short_word_pair_hull_top_words_per_color,
        short_word_pair_hull_workers=args.short_word_pair_hull_workers,
        short_word_pair_hull_batch_size=args.short_word_pair_hull_batch_size,
        short_word_pair_hull_min_fractional_shared_colors=args.short_word_pair_hull_min_fractional_shared_colors,
    )


def base_vocabulary() -> list[str]:
    seen = set()
    result = []
    for token in [*DEFAULT_SPECIAL_TOKENS, *byte_level_alphabet()]:
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def load_incumbent(args: argparse.Namespace, output_dir: Path, eval_texts: list[str]) -> Incumbent:
    incumbent = Incumbent()
    if args.incumbent_tokenizer:
        source_path = Path(args.incumbent_tokenizer)
        tokenizer = LpDpTokenizer.from_file(source_path)
        stats = evaluate_texts("branch-cut incumbent", tokenizer, eval_texts, num_workers=args.eval_workers)
        incumbent.tokenizer = tokenizer
        incumbent.stats = stats
        incumbent.path = output_dir / "best_tokenizer.json"
        tokenizer.save(incumbent.path)
        write_best_metadata(
            output_dir,
            {
                "source": str(source_path),
                "node": None,
                "tokens": stats.tokens,
                "bytes": stats.bytes,
                "bytes_per_token": stats.bytes_per_token,
                "tokens_per_byte": stats.tokens_per_byte,
                "bound": None,
                "gap_fraction": None,
                "fixes": [],
            },
        )
        LOGGER.info("Seeded incumbent from %s tokens=%d", source_path, stats.tokens)
    if args.incumbent_tokens is not None:
        incumbent.bound = float(args.incumbent_tokens)
        if incumbent.stats is None:
            incumbent.stats = CompressionStats(
                name="known incumbent",
                files=0,
                bytes=0,
                chars=0,
                tokens=int(args.incumbent_tokens),
            )
        LOGGER.info("Seeded incumbent token count from CLI: %s", format_bound(float(args.incumbent_tokens)))
    return incumbent


def solve_with_global_cuts(
    solver: HighsWarmLpSolver,
    lp,
    *,
    existing_cut_keys: set[tuple],
    rounds: int,
    cuts_per_round: int,
    tolerance: float,
    families: tuple[str, ...],
    config: CutSeparationConfig,
    label: str,
):
    solution = None
    total_added = 0
    for iteration in range(rounds + 1):
        solve_start = time.monotonic()
        solution = solver.solve()
        LOGGER.info(
            "%s cut iteration %d solved in %.3fs objective=%s",
            label,
            iteration,
            time.monotonic() - solve_start,
            format_bound(float(solution.fun)) if solution.success else solution.message,
        )
        if not solution.success or iteration == rounds:
            return solution, total_added
        cut_matrix, cut_rhs, cut_keys, max_violation = separate_cuts(
            lp,
            solution.x,
            existing_cut_keys=existing_cut_keys,
            max_cuts=cuts_per_round,
            tolerance=tolerance,
            families=families,
            config=config,
        )
        LOGGER.info(
            "%s cut iteration %d: added_cuts=%d max_violation=%.6g",
            label,
            iteration,
            len(cut_keys),
            max_violation,
        )
        if not cut_keys:
            return solution, total_added
        existing_cut_keys.update(cut_keys)
        solver.add_ub_rows(cut_matrix, cut_rhs)
        total_added += len(cut_keys)
    return solution, total_added


def count_fractional_colors(lp, x_values) -> int:
    t_offset = lp["num_nonfree_edges"] + lp["num_free_edges"]
    return sum(1 for value in x_values[t_offset:] if 1e-6 < value < 1.0 - 1e-6)


def round_node_solution(
    *,
    tokens,
    lp,
    solution_x,
    fixed: dict[int, int],
    base_vocab: list[str],
    token_budget: int,
    pretokenizer_mode: str,
) -> LpDpTokenizer | None:
    fixed_one = [idx for idx, value in sorted(fixed.items()) if value == 1]
    if len(fixed_one) > token_budget:
        LOGGER.info("Skipping rounded node: %d fixed-in tokens exceed budget %d", len(fixed_one), token_budget)
        return None

    fixed_one_tokens = []
    seen_vocab = set(base_vocab)
    for token_idx in fixed_one:
        token = tokens[token_idx].token
        if token not in seen_vocab:
            fixed_one_tokens.append(token)
            seen_vocab.add(token)

    excluded = set(base_vocab)
    excluded.update(tokens[idx].token for idx, value in fixed.items() if value == 0)
    excluded.update(fixed_one_tokens)
    candidates = candidates_from_solution(tokens, lp, solution_x)
    remaining_budget = token_budget - len(fixed_one_tokens)
    selected = round_lp_tokens(candidates, remaining_budget, excluded=excluded)
    vocab = [*base_vocab, *fixed_one_tokens, *(token.token for token in selected)]
    return LpDpTokenizer(vocab, pretokenizer_mode=pretokenizer_mode, unk_token=DEFAULT_UNK_TOKEN)


def evaluate_node_tokenizer(
    node_id: int,
    tokenizer: LpDpTokenizer,
    eval_texts: list[str],
    *,
    eval_workers: int,
    output_dir: Path,
    save_node_tokenizers: bool,
) -> CompressionStats:
    stats = evaluate_texts(f"branch-cut node {node_id} rounded", tokenizer, eval_texts, num_workers=eval_workers)
    if save_node_tokenizers:
        node_dir = output_dir / "nodes"
        node_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(node_dir / f"node_{node_id:05d}_tokenizer.json")
    return stats


def save_best_incumbent(
    incumbent: Incumbent,
    *,
    tokenizer: LpDpTokenizer,
    stats: CompressionStats,
    output_dir: Path,
    node_id: int,
    bound: float,
    fixed: dict[int, int],
) -> None:
    best_path = output_dir / "best_tokenizer.json"
    tokenizer.save(best_path)
    incumbent.tokenizer = tokenizer
    incumbent.stats = stats
    incumbent.path = best_path
    incumbent.node_id = node_id
    incumbent.bound = bound
    metadata = {
        "node": node_id,
        "tokens": stats.tokens,
        "bytes": stats.bytes,
        "bytes_per_token": stats.bytes_per_token,
        "tokens_per_byte": stats.tokens_per_byte,
        "bound": bound,
        "gap_fraction": (stats.tokens - bound) / stats.tokens if stats.tokens else 0.0,
        "fixes": [[int(k), int(v)] for k, v in fixed.items()],
    }
    write_best_metadata(output_dir, metadata)
    LOGGER.info(
        "Saved new best branch-cut tokenizer: node=%d tokens=%d bound=%s path=%s",
        node_id,
        stats.tokens,
        format_bound(bound),
        best_path,
    )


def write_best_metadata(output_dir: Path, metadata: dict) -> None:
    (output_dir / "best_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def write_search_metadata(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    processed: int,
    pruned: int,
    infeasible: int,
    queue: list[BranchNode],
    terminal_bounds: list[float],
    root_bound: float,
    incumbent: Incumbent,
    node_metadata: list[dict],
    elapsed: float,
) -> None:
    certified_bound = min([*terminal_bounds, *(node.parent_bound for node in queue)], default=root_bound)
    payload = {
        "args": vars(args),
        "processed": processed,
        "pruned": pruned,
        "infeasible": infeasible,
        "remaining": len(queue),
        "root_bound": root_bound,
        "certified_bound": certified_bound,
        "best_tokens": None if not math.isfinite(incumbent.tokens) else incumbent.tokens,
        "best_node": incumbent.node_id,
        "best_path": str(incumbent.path) if incumbent.path is not None else None,
        "elapsed_seconds": elapsed,
        "nodes": node_metadata,
    }
    metadata_path = output_dir / "search_metadata.json"
    tmp_path = metadata_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    shutil.move(tmp_path, metadata_path)


if __name__ == "__main__":
    main()
