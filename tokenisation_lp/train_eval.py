from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tokenisation_lp.bpe_training import train_bpe_tokenizer
from tokenisation_lp.corpus import load_texts
from tokenisation_lp.evaluation import evaluate_texts
from tokenisation_lp.lp_training import CutSeparationConfig, train_lp_tokenizer
from tokenisation_lp.pretokenization import DEFAULT_SPECIAL_TOKENS, DEFAULT_UNK_TOKEN


LOGGER = logging.getLogger(__name__)


def train_and_eval_lp(
    *,
    train_texts: list[str],
    eval_texts: list[str],
    vocab_size: int,
    output_dir: Path,
    pretokenizer_mode: str,
    min_token_count: int,
    max_token_length: int | None,
    eval_workers: int,
    cut_rounds: int,
    cuts_per_round: int,
    cut_tolerance: float,
    cut_families: tuple[str, ...],
    cut_config: CutSeparationConfig,
    lp_solution_cache_dir: str | None,
    lp_solver: str,
):
    saw_iteration = False
    best_tokens = None
    best_iteration = None
    lp_output_dir = output_dir / "lp"
    iteration_dir = lp_output_dir / "iterations"

    def on_iteration(iteration, tokenizer):
        nonlocal saw_iteration, best_tokens, best_iteration
        saw_iteration = True
        log_lp_relaxation_bound_value(
            f"LP iteration {iteration.iteration} relaxation bound",
            iteration.token_count_lower_bound,
            eval_texts,
        )
        stats = evaluate_texts(
            f"lp iteration {iteration.iteration} rounded",
            tokenizer,
            eval_texts,
            num_workers=eval_workers,
        )
        gap_fraction = (
            (stats.tokens - iteration.token_count_lower_bound) / stats.tokens
            if stats.tokens
            else 0.0
        )
        LOGGER.info(
            "LP iteration %d rounded gap: actual_tokens=%d lower_bound=%.3f gap=%.4f%%",
            iteration.iteration,
            stats.tokens,
            iteration.token_count_lower_bound,
            100.0 * gap_fraction,
        )
        iteration_dir.mkdir(parents=True, exist_ok=True)
        iteration_path = iteration_dir / f"lp_iteration_{iteration.iteration:03d}.json"
        tokenizer.save(iteration_path)
        LOGGER.info("Saved LP iteration %d rounded tokenizer to %s", iteration.iteration, iteration_path)
        is_best = best_tokens is None or stats.tokens < best_tokens
        if is_best:
            best_tokens = stats.tokens
            best_iteration = iteration.iteration
            best_path = lp_output_dir / "best_so_far_tokenizer.json"
            tokenizer.save(best_path)
            LOGGER.info(
                "Saved new best LP rounded tokenizer: iteration=%d tokens=%d path=%s",
                iteration.iteration,
                stats.tokens,
                best_path,
            )
        metadata = {
            "iteration": iteration.iteration,
            "actual_tokens": stats.tokens,
            "lower_bound": iteration.token_count_lower_bound,
            "gap_fraction": gap_fraction,
            "fractional_colors": iteration.fractional_colors,
            "active_cuts": iteration.total_cuts,
            "next_cuts": iteration.added_cuts,
            "max_cut_violation": iteration.max_violation,
            "best_iteration": best_iteration,
            "best_tokens": best_tokens,
        }
        (iteration_dir / f"lp_iteration_{iteration.iteration:03d}_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        (lp_output_dir / "best_so_far_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    result = train_lp_tokenizer(
        train_texts,
        vocab_size,
        pretokenizer_mode=pretokenizer_mode,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        unk_token=DEFAULT_UNK_TOKEN,
        min_token_count=min_token_count,
        max_token_length=max_token_length,
        output_dir=lp_output_dir,
        cut_rounds=cut_rounds,
        cuts_per_round=cuts_per_round,
        cut_tolerance=cut_tolerance,
        cut_families=cut_families,
        cut_config=cut_config,
        lp_solution_cache_dir=lp_solution_cache_dir,
        lp_solver=lp_solver,
        iteration_callback=on_iteration,
    )
    if not saw_iteration:
        log_lp_relaxation_bound(result, eval_texts)
        evaluate_texts("lp", result.tokenizer, eval_texts, num_workers=eval_workers)
    return result


def log_lp_relaxation_bound(result, texts: list[str]) -> None:
    log_lp_relaxation_bound_value(
        "LP relaxation bound",
        result.corpus_token_count_lower_bound,
        texts,
    )


def log_lp_relaxation_bound_value(label: str, lower_bound: float, texts: list[str]) -> None:
    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    if lower_bound <= 0:
        return
    LOGGER.info(
        "%s: tokens>=%.3f bytes/token<=%.4f tokens/byte>=%.6f",
        label,
        lower_bound,
        total_bytes / lower_bound,
        lower_bound / total_bytes if total_bytes else 0.0,
    )


def train_and_eval_bpe(
    *,
    train_texts: list[str],
    eval_texts: list[str],
    vocab_size: int,
    output_dir: Path,
    pretokenizer_mode: str,
):
    result = train_bpe_tokenizer(
        train_texts,
        vocab_size,
        pretokenizer_mode=pretokenizer_mode,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        unk_token=DEFAULT_UNK_TOKEN,
        output_dir=output_dir / "bpe",
    )
    evaluate_texts("bpe", result.tokenizer, eval_texts)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LP and/or greedy BPE tokenizers from a directory of text files, then evaluate compression."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing training text files.")
    parser.add_argument(
        "--eval-dir",
        help="Optional directory containing evaluation text files. Defaults to --data-dir.",
    )
    parser.add_argument("--vocab-size", required=True, type=int, help="Target vocabulary size.")
    parser.add_argument(
        "--kind",
        choices=("lp", "bpe", "both"),
        default="both",
        help="Which tokenizer family to train.",
    )
    parser.add_argument(
        "--output-dir",
        default="tokenizer_runs",
        help="Directory where tokenizer JSON files are written.",
    )
    parser.add_argument(
        "--pretokenizer",
        default="bytelevel",
        choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"),
        help="Pretokenizer used by both LP and BPE.",
    )
    parser.add_argument(
        "--min-token-count",
        type=int,
        default=2,
        help="Minimum corpus occurrence count for LP multi-character token candidates.",
    )
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=16,
        help="Maximum LP candidate token length. Use 0 to consider every substring.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="Number of worker processes for LP DP evaluation.",
    )
    parser.add_argument(
        "--lp-cut-rounds",
        type=int,
        default=0,
        help="Number of same-colour byte-boundary cut rounds to add after the initial LP solve.",
    )
    parser.add_argument(
        "--lp-cuts-per-round",
        type=int,
        default=100,
        help="Maximum number of violated byte-boundary cuts to add per LP round.",
    )
    parser.add_argument(
        "--lp-cut-tolerance",
        type=float,
        default=1e-6,
        help="Minimum violation required to add a byte-boundary cut.",
    )
    parser.add_argument(
        "--lp-cut-families",
        default="boundary",
        help=(
            "Comma-separated LP cut families. Supported: "
            "boundary,word_packing,global_token_packing,global_pair_packing,global_triple_packing,"
            "global_rank_count,word_rank_count,word_rank_length,path_config,path_multicover,path_min_cover,group_value,"
            "threshold_value,group_budget_value,word_hull,short_word_hull,short_word_full_hull,"
            "short_word_pair_hull,group_value_deep,"
            "conflict_clique,conflict_odd_cycle,word_support,bad_vocab_escape,"
            "bad_vocab_improvement,window_overlap,window_overlap_deep,word_path_cover,window_pair."
        ),
    )
    parser.add_argument(
        "--lp-solution-cache-dir",
        default=None,
        help="Optional directory for caching identical LP solutions, e.g. /tmp/tokenizer_lp_cache.",
    )
    parser.add_argument(
        "--lp-solver",
        choices=("highspy", "scipy"),
        default="highspy",
        help="LP solver backend. highspy keeps a simplex model alive for iterative cut warm starts.",
    )
    parser.add_argument(
        "--lp-word-support-max-words",
        type=int,
        default=2000,
        help="Maximum suspicious word types scanned by the word_support separator.",
    )
    parser.add_argument(
        "--lp-word-support-max-rank",
        type=int,
        default=20,
        help="Maximum fractional token colours selected per word_support cut.",
    )
    parser.add_argument(
        "--lp-word-support-max-paths",
        type=int,
        default=100000,
        help="Maximum exact segmentation paths enumerated per word_support candidate.",
    )
    parser.add_argument(
        "--lp-short-word-hull-max-words",
        type=int,
        default=2000,
        help="Maximum short word types scanned by the short_word_hull separator.",
    )
    parser.add_argument(
        "--lp-short-word-hull-max-length",
        type=int,
        default=10,
        help="Maximum pretokenized byte length scanned by the short_word_hull separator.",
    )
    parser.add_argument(
        "--lp-short-word-hull-max-rank",
        type=int,
        default=16,
        help="Maximum fractional token colours selected per short_word_hull cut.",
    )
    parser.add_argument(
        "--lp-short-word-full-hull-max-words",
        type=int,
        default=250,
        help="Maximum frequent short word types scanned by the short_word_full_hull separator.",
    )
    parser.add_argument(
        "--lp-short-word-full-hull-max-length",
        type=int,
        default=8,
        help="Maximum pretokenized byte length scanned by the short_word_full_hull separator.",
    )
    parser.add_argument(
        "--lp-short-word-full-hull-max-colors",
        type=int,
        default=64,
        help="Maximum local token colours allowed for short_word_full_hull; larger words are skipped.",
    )
    parser.add_argument(
        "--lp-short-word-pair-hull-max-words",
        type=int,
        default=500,
        help="Maximum short word types used to propose short_word_pair_hull pairs.",
    )
    parser.add_argument(
        "--lp-short-word-pair-hull-max-length",
        type=int,
        default=12,
        help="Maximum pretokenized byte length used by the short_word_pair_hull separator.",
    )
    parser.add_argument(
        "--lp-short-word-pair-hull-max-colors",
        type=int,
        default=96,
        help="Maximum union of local token colours allowed for a short_word_pair_hull pair.",
    )
    parser.add_argument(
        "--lp-short-word-pair-hull-max-pair-rows",
        type=int,
        default=250000,
        help="Maximum path-pair constraints allowed for one short_word_pair_hull separator LP.",
    )
    parser.add_argument(
        "--lp-short-word-pair-hull-max-pairs",
        type=int,
        default=800,
        help="Maximum candidate word pairs tested by short_word_pair_hull per separation round.",
    )
    parser.add_argument(
        "--lp-short-word-pair-hull-top-words-per-color",
        type=int,
        default=36,
        help="Number of words per fractional colour used to propose short_word_pair_hull candidates.",
    )
    parser.add_argument(
        "--lp-short-word-pair-hull-workers",
        type=int,
        default=0,
        help="Worker processes for short_word_pair_hull separation. Use 0 for cpu_count-1.",
    )
    parser.add_argument(
        "--lp-short-word-pair-hull-batch-size",
        type=int,
        default=32,
        help="Candidate pairs submitted per worker task for short_word_pair_hull multiprocessing.",
    )
    parser.add_argument(
        "--lp-short-word-pair-hull-min-fractional-shared-colors",
        type=int,
        default=1,
        help=(
            "Minimum number of shared fractional token colors required before testing a "
            "short_word_pair_hull candidate."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    train_texts = load_texts(args.data_dir)
    eval_texts = load_texts(args.eval_dir or args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_token_length = None if args.max_token_length == 0 else args.max_token_length

    LOGGER.info(
        "Loaded %d training files and %d evaluation files",
        len(train_texts),
        len(eval_texts),
    )
    cut_families = tuple(
        family.strip()
        for family in args.lp_cut_families.split(",")
        if family.strip()
    )
    cut_config = CutSeparationConfig(
        word_support_max_words=args.lp_word_support_max_words,
        word_support_max_rank=args.lp_word_support_max_rank,
        word_support_max_paths=args.lp_word_support_max_paths,
        short_word_hull_max_words=args.lp_short_word_hull_max_words,
        short_word_hull_max_length=args.lp_short_word_hull_max_length,
        short_word_hull_max_rank=args.lp_short_word_hull_max_rank,
        short_word_full_hull_max_words=args.lp_short_word_full_hull_max_words,
        short_word_full_hull_max_length=args.lp_short_word_full_hull_max_length,
        short_word_full_hull_max_colors=args.lp_short_word_full_hull_max_colors,
        short_word_pair_hull_max_words=args.lp_short_word_pair_hull_max_words,
        short_word_pair_hull_max_length=args.lp_short_word_pair_hull_max_length,
        short_word_pair_hull_max_colors=args.lp_short_word_pair_hull_max_colors,
        short_word_pair_hull_max_pair_rows=args.lp_short_word_pair_hull_max_pair_rows,
        short_word_pair_hull_max_pairs=args.lp_short_word_pair_hull_max_pairs,
        short_word_pair_hull_top_words_per_color=args.lp_short_word_pair_hull_top_words_per_color,
        short_word_pair_hull_workers=args.lp_short_word_pair_hull_workers,
        short_word_pair_hull_batch_size=args.lp_short_word_pair_hull_batch_size,
        short_word_pair_hull_min_fractional_shared_colors=args.lp_short_word_pair_hull_min_fractional_shared_colors,
    )

    if args.kind in {"lp", "both"}:
        train_and_eval_lp(
            train_texts=train_texts,
            eval_texts=eval_texts,
            vocab_size=args.vocab_size,
            output_dir=output_dir,
            pretokenizer_mode=args.pretokenizer,
            min_token_count=args.min_token_count,
            max_token_length=max_token_length,
            eval_workers=args.eval_workers,
            cut_rounds=args.lp_cut_rounds,
            cuts_per_round=args.lp_cuts_per_round,
            cut_tolerance=args.lp_cut_tolerance,
            cut_families=cut_families,
            cut_config=cut_config,
            lp_solution_cache_dir=args.lp_solution_cache_dir,
            lp_solver=args.lp_solver,
        )

    if args.kind in {"bpe", "both"}:
        train_and_eval_bpe(
            train_texts=train_texts,
            eval_texts=eval_texts,
            vocab_size=args.vocab_size,
            output_dir=output_dir,
            pretokenizer_mode=args.pretokenizer,
        )


if __name__ == "__main__":
    main()
