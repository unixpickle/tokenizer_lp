from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tokenisation_lp.bpe_training import train_bpe_tokenizer
from tokenisation_lp.corpus import load_texts
from tokenisation_lp.evaluation import evaluate_texts
from tokenisation_lp.lp_training import train_lp_tokenizer
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
    lp_solution_cache_dir: str | None,
    lp_solver: str,
):
    saw_iteration = False

    def on_iteration(iteration, tokenizer):
        nonlocal saw_iteration
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

    result = train_lp_tokenizer(
        train_texts,
        vocab_size,
        pretokenizer_mode=pretokenizer_mode,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        unk_token=DEFAULT_UNK_TOKEN,
        min_token_count=min_token_count,
        max_token_length=max_token_length,
        output_dir=output_dir / "lp",
        cut_rounds=cut_rounds,
        cuts_per_round=cuts_per_round,
        cut_tolerance=cut_tolerance,
        cut_families=cut_families,
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
            "path_config,path_multicover,"
            "window_overlap,window_overlap_deep,word_path_cover,window_pair."
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
