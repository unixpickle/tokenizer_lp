from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from tokenisation_lp.alternative_pair_diagnostics import (
    WordProfile,
    affix_similarity,
    build_word_profiles,
    cosine_score,
    jaccard_score,
    normalize_strategy_score,
    pair_key,
    propose_candidates as propose_pair_candidates,
)
from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    HighsWarmLpSolver,
    build_standard_form,
    count_pretokenized_strings,
    prepare_lp_data,
)
from tokenisation_lp.pair_hull_diagnostics import apply_individual_hulls
from tokenisation_lp.pretokenization import (
    DEFAULT_SPECIAL_TOKENS,
    build_pretokenizer,
    byte_level_alphabet,
)
from tokenisation_lp.subset_hull_diagnostics import (
    SubsetCut,
    SubsetHullContext,
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
    solver = HighsWarmLpSolver(
        c=lp["c"],
        A_ub=lp["A_ub"],
        b_ub=b_ub,
        A_eq=lp["A_eq"],
        b_eq=lp["b_eq"],
        lb=lp["lb"],
        ub=lp["ub"],
        cache_dir=args.lp_solution_cache_dir,
    )
    solution = solver.solve()
    if not solution.success:
        raise RuntimeError(f"LP solve failed: {solution.message}")
    LOGGER.info("Base LP bound %.3f", solution.fun)
    if args.apply_individual_hulls:
        solution = apply_individual_hulls(args, lp, solver, solution)

    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    f_values = solution.x[:num_f]
    g_values = solution.x[num_f : num_f + num_g]
    t_values = solution.x[num_f + num_g :]
    profiles = build_word_profiles(lp, words, freqs, f_values, t_values, args)
    LOGGER.info("Built %d fractional word profiles", len(profiles))

    context = SubsetHullContext(
        lp=lp,
        words=words,
        freqs=freqs,
        f_values=f_values,
        g_values=g_values,
        t_values=t_values,
        max_paths=args.max_paths,
        max_colors=args.max_colors,
        max_product_rows=args.max_product_rows,
        tolerance=args.cut_tolerance,
    )

    candidates = propose_triplets(lp, f_values, t_values, profiles, args)
    triplets, metadata = merge_triplets(candidates, args.max_total_triplets)
    LOGGER.info(
        "Merged triplet candidates: total=%d capped=%d strategies=%s",
        len(metadata),
        len(triplets),
        ", ".join(sorted(candidates)),
    )

    start = time.monotonic()
    rows, skipped = test_triplets(context, triplets, metadata, args)
    elapsed = time.monotonic() - start
    cuts = [row for row in rows if row["has_cut"]]
    LOGGER.info(
        "Triplet diagnostics done: tested=%d cuts=%d pair_clean_cuts=%d elapsed=%.3fs skipped=%s",
        len(rows),
        len(cuts),
        sum(1 for row in cuts if not row["pair_explained"]),
        elapsed,
        dict(skipped),
    )
    report = {
        "run": {
            "data_dir": str(Path(args.data_dir).expanduser()),
            "vocab_size": args.vocab_size,
            "pretokenizer": args.pretokenizer,
            "lp_bound": float(solution.fun),
            "elapsed_seconds": elapsed,
            "max_total_triplets": args.max_total_triplets,
            "max_test_triplets": args.max_test_triplets,
            "max_product_rows": args.max_product_rows,
        },
        "summary": summarize_rows(rows),
        "strategy_summary": strategy_summary(rows, candidates),
        "examples": {
            "cuts_by_violation": sorted(cuts, key=lambda row: row["violation"], reverse=True)[: args.examples_per_class],
            "pair_clean_cuts_by_violation": sorted(
                [row for row in cuts if not row["pair_explained"]],
                key=lambda row: row["violation"],
                reverse=True,
            )[: args.examples_per_class],
            "no_cuts_by_score": sorted(
                [row for row in rows if not row["has_cut"]],
                key=lambda row: row["score"],
                reverse=True,
            )[: args.examples_per_class],
        },
        "skipped": dict(skipped),
    }
    if args.dump_rows:
        report["rows"] = rows

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    LOGGER.info("Wrote triplet report: %s", output_path)
    for item in report["strategy_summary"]:
        LOGGER.info(
            "%s proposed=%d tested=%d cuts=%d pair_clean=%d precision=%.4f",
            item["strategy"],
            item["proposed"],
            item["tested"],
            item["cuts"],
            item["pair_clean_cuts"],
            item["precision"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore word-triplet projected hull violations.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8)
    parser.add_argument("--lp-solution-cache-dir", default="/tmp/tokenizer_lp_solution_cache")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--apply-individual-hulls", action="store_true")
    parser.add_argument("--individual-rounds", type=int, default=3)
    parser.add_argument("--individual-max-words", type=int, default=12000)
    parser.add_argument("--individual-max-length", type=int, default=12)
    parser.add_argument("--individual-max-colors", type=int, default=96)
    parser.add_argument("--max-words", type=int, default=700)
    parser.add_argument("--max-word-length", type=int, default=12)
    parser.add_argument("--max-colors", type=int, default=128)
    parser.add_argument("--max-product-rows", type=int, default=200000)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--cut-tolerance", type=float, default=1e-6)
    parser.add_argument("--top-colors", type=int, default=60)
    parser.add_argument("--top-words-per-color", type=int, default=8)
    parser.add_argument("--neighbors-per-word", type=int, default=8)
    parser.add_argument("--top-colors-per-word", type=int, default=10)
    parser.add_argument("--max-candidates-per-strategy", type=int, default=600)
    parser.add_argument("--max-total-triplets", type=int, default=1500)
    parser.add_argument("--max-test-triplets", type=int, default=500)
    parser.add_argument("--classify-pair-explained", action="store_true")
    parser.add_argument("--skip-pair-explained", action="store_true")
    parser.add_argument("--examples-per-class", type=int, default=20)
    parser.add_argument("--dump-rows", action="store_true")
    parser.add_argument("--output", default="/tmp/tokenizer_lp_triplet_hull_diagnostics.json")
    # Imported pair candidate generator expects these names.
    parser.set_defaults(max_pairs=8000, max_pair_rows=250000, workers=0, batch_size=64)
    return parser.parse_args()


def propose_triplets(lp, f_values, t_values, profiles: dict[int, WordProfile], args):
    return {
        "shared_color_triples": shared_color_triples(t_values, profiles, args),
        "top_color_cluster": top_color_cluster(profiles, args),
        "surface_cluster": surface_cluster(profiles, args),
        "pair_extension": pair_extension(lp, f_values, t_values, profiles, args),
    }


def shared_color_triples(t_values, profiles, args):
    color_rows = defaultdict(list)
    for profile in profiles.values():
        for token_idx, score in profile.color_scores.items():
            token_value = float(t_values[token_idx])
            if args.cut_tolerance < token_value < 1.0 - args.cut_tolerance:
                weighted = score * min(token_value, 1.0 - token_value) * math.sqrt(max(1.0, profile.freq))
                color_rows[token_idx].append((weighted, profile.word_idx))
    for rows in color_rows.values():
        rows.sort(reverse=True)

    triplet_scores = defaultdict(float)
    for rows in color_rows.values():
        for combo in combinations(rows[: args.top_words_per_color], 3):
            words = tuple(sorted(word_idx for _, word_idx in combo))
            triplet_scores[words] += min(score for score, _ in combo)
    return top_triplets(triplet_scores, args.max_candidates_per_strategy)


def top_color_cluster(profiles, args):
    triplet_scores = defaultdict(float)
    for anchor in profiles.values():
        scored = []
        for other in profiles.values():
            if other.word_idx == anchor.word_idx:
                continue
            score = 0.6 * jaccard_score(anchor.top_colors, other.top_colors) + 0.4 * cosine_score(anchor, other)
            if score <= 0:
                continue
            impact = math.sqrt(anchor.weighted_score * other.weighted_score)
            scored.append((score * impact, other.word_idx))
        neighbors = sorted(scored, reverse=True)[: args.neighbors_per_word]
        for (left_score, left), (right_score, right) in combinations(neighbors, 2):
            words = tuple(sorted((anchor.word_idx, left, right)))
            if len(set(words)) != 3:
                continue
            triplet_scores[words] = max(triplet_scores[words], min(left_score, right_score))
    return top_triplets(triplet_scores, args.max_candidates_per_strategy)


def surface_cluster(profiles, args):
    triplet_scores = defaultdict(float)
    for anchor in profiles.values():
        scored = []
        for other in profiles.values():
            if other.word_idx == anchor.word_idx:
                continue
            ngram = jaccard_score(anchor.ngrams, other.ngrams)
            affix = affix_similarity(anchor.normalized_word, other.normalized_word)
            score = max(ngram, affix)
            if score <= 0:
                continue
            shared_color_bonus = 1.0 + jaccard_score(anchor.top_colors, other.top_colors)
            impact = math.sqrt(anchor.weighted_score * other.weighted_score)
            scored.append((score * shared_color_bonus * impact, other.word_idx))
        neighbors = sorted(scored, reverse=True)[: args.neighbors_per_word]
        for (left_score, left), (right_score, right) in combinations(neighbors, 2):
            words = tuple(sorted((anchor.word_idx, left, right)))
            if len(set(words)) != 3:
                continue
            triplet_scores[words] = max(triplet_scores[words], min(left_score, right_score))
    return top_triplets(triplet_scores, args.max_candidates_per_strategy)


def pair_extension(lp, f_values, t_values, profiles, args):
    pair_candidates = propose_pair_candidates(lp, f_values, t_values, profiles, args)
    pair_scores = {}
    for strategy, pairs in pair_candidates.items():
        for key, score in pairs.items():
            pair_scores[key] = max(pair_scores.get(key, 0.0), normalize_strategy_score(strategy, score))
    ranked_pairs = sorted(pair_scores.items(), key=lambda item: item[1], reverse=True)[: args.max_candidates_per_strategy]

    triplet_scores = defaultdict(float)
    profile_values = list(profiles.values())
    for (left, right), pair_score in ranked_pairs:
        left_profile = profiles.get(left)
        right_profile = profiles.get(right)
        if left_profile is None or right_profile is None:
            continue
        pair_colors = left_profile.top_colors | right_profile.top_colors
        scored = []
        for third in profile_values:
            if third.word_idx in {left, right}:
                continue
            color_score = jaccard_score(pair_colors, third.top_colors)
            surface_score = max(
                affix_similarity(left_profile.normalized_word, third.normalized_word),
                affix_similarity(right_profile.normalized_word, third.normalized_word),
                jaccard_score(left_profile.ngrams, third.ngrams),
                jaccard_score(right_profile.ngrams, third.ngrams),
            )
            score = max(color_score, surface_score)
            if score <= 0:
                continue
            scored.append((pair_score * score * math.sqrt(max(1.0, third.freq)), third.word_idx))
        for third_score, third in sorted(scored, reverse=True)[: max(2, args.neighbors_per_word // 2)]:
            words = tuple(sorted((left, right, third)))
            triplet_scores[words] = max(triplet_scores[words], third_score)
    return top_triplets(triplet_scores, args.max_candidates_per_strategy)


def merge_triplets(candidates_by_strategy, max_total_triplets):
    metadata = {}
    for strategy, triplets in candidates_by_strategy.items():
        for key, score in triplets.items():
            record = metadata.setdefault(key, {"strategies": set(), "strategy_scores": {}, "score": 0.0})
            record["strategies"].add(strategy)
            record["strategy_scores"][strategy] = float(score)
            record["score"] = max(record["score"], normalize_strategy_score(strategy, score))
    ranked = sorted(metadata.items(), key=lambda item: item[1]["score"], reverse=True)[:max_total_triplets]
    return [(data["score"], key) for key, data in ranked], {key: data for key, data in ranked}


def test_triplets(context: SubsetHullContext, triplets, metadata, args):
    rows = []
    skipped = defaultdict(int)
    pair_cut_cache = {}
    for rank, (score, triplet) in enumerate(triplets[: args.max_test_triplets], start=1):
        pair_explained = False
        pair_cut_words = []
        if args.classify_pair_explained or args.skip_pair_explained:
            for pair in combinations(triplet, 2):
                pair = tuple(sorted(pair))
                if pair not in pair_cut_cache:
                    pair_cut_cache[pair] = context.separate_subset(pair)
                if isinstance(pair_cut_cache[pair], SubsetCut):
                    pair_explained = True
                    pair_cut_words.append(pair)
            if args.skip_pair_explained and pair_explained:
                skipped["pair_explained"] += 1
                continue

        result = context.separate_subset(triplet)
        if result is None:
            skipped["no_cut"] += 1
            rows.append(row_for_result(context, rank, score, triplet, metadata, None, pair_explained, pair_cut_words))
            continue
        if isinstance(result, str):
            skipped[result] += 1
            continue
        rows.append(row_for_result(context, rank, score, triplet, metadata, result, pair_explained, pair_cut_words))
    return rows, skipped


def row_for_result(context, rank, score, triplet, metadata, cut, pair_explained, pair_cut_words):
    data = metadata[triplet]
    row = {
        "rank": rank,
        "score": float(score),
        "word_indices": list(triplet),
        "words": [context.words[idx] for idx in triplet],
        "lengths": [int(context.lp["word_lengths"][idx]) for idx in triplet],
        "freqs": [int(context.freqs[idx]) for idx in triplet],
        "strategies": sorted(data["strategies"]),
        "strategy_scores": {key: float(value) for key, value in sorted(data["strategy_scores"].items())},
        "pair_explained": bool(pair_explained),
        "pair_cut_words": [[context.words[left], context.words[right]] for left, right in pair_cut_words],
    }
    if cut is None:
        row.update({"has_cut": False, "violation": 0.0})
    else:
        row.update(
            {
                "has_cut": True,
                "violation": float(cut.violation),
                "num_colors": int(cut.num_colors),
                "num_rows": int(cut.num_rows),
                "build_seconds": float(cut.build_seconds),
                "solve_seconds": float(cut.solve_seconds),
            }
        )
    return row


def strategy_summary(rows, candidates_by_strategy):
    output = []
    for strategy, proposed in sorted(candidates_by_strategy.items()):
        tested = [row for row in rows if strategy in row["strategies"]]
        cuts = [row for row in tested if row["has_cut"]]
        pair_clean = [row for row in cuts if not row["pair_explained"]]
        output.append(
            {
                "strategy": strategy,
                "proposed": len(proposed),
                "tested": len(tested),
                "cuts": len(cuts),
                "pair_clean_cuts": len(pair_clean),
                "precision": len(cuts) / max(1, len(tested)),
                "pair_clean_precision": len(pair_clean) / max(1, len(tested)),
                "max_violation": max((row["violation"] for row in cuts), default=0.0),
            }
        )
    return output


def summarize_rows(rows):
    cuts = [row for row in rows if row["has_cut"]]
    pair_clean = [row for row in cuts if not row["pair_explained"]]
    return {
        "tested": len(rows),
        "cuts": len(cuts),
        "pair_clean_cuts": len(pair_clean),
        "cut_rate": len(cuts) / max(1, len(rows)),
        "pair_clean_cut_rate": len(pair_clean) / max(1, len(rows)),
        "max_violation": max((row["violation"] for row in cuts), default=0.0),
    }


def top_triplets(triplet_scores, limit):
    return dict(sorted(triplet_scores.items(), key=lambda item: item[1], reverse=True)[:limit])


if __name__ == "__main__":
    main()
