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

from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    HighsWarmLpSolver,
    build_standard_form,
    count_pretokenized_strings,
    prepare_lp_data,
    rank_short_fractional_words,
    short_word_pair_candidates,
)
from tokenisation_lp.pair_hull_diagnostics import (
    apply_individual_hulls,
    build_report,
    enrich_result,
    prepare_pair_tasks,
    run_pair_diagnostics,
)
from tokenisation_lp.pretokenization import (
    DEFAULT_SPECIAL_TOKENS,
    build_pretokenizer,
    byte_level_alphabet,
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
    candidates = propose_candidates(lp, f_values, t_values, profiles, args)
    pair_rows, metadata = merge_candidates(candidates, args.max_total_pairs)
    LOGGER.info(
        "Merged candidates: total=%d capped=%d strategies=%s",
        len(metadata),
        len(pair_rows),
        ", ".join(sorted(candidates)),
    )

    start = time.monotonic()
    tasks, paths_by_word, skipped, pattern_count = prepare_pair_tasks(
        lp,
        pair_rows,
        max_pairs=len(pair_rows),
        max_colors=args.max_colors,
        max_pair_rows=args.max_pair_rows,
        max_paths=args.max_paths,
    )
    LOGGER.info(
        "Prepared alternative pair tasks: tasks=%d skipped=%s patterns=%d",
        len(tasks),
        dict(skipped),
        pattern_count,
    )
    results = run_pair_diagnostics(
        lp,
        f_values,
        g_values,
        t_values,
        paths_by_word,
        tasks,
        tolerance=args.cut_tolerance,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    elapsed = time.monotonic() - start

    word_color_scores = defaultdict(dict, {
        word_idx: profile.color_scores
        for word_idx, profile in profiles.items()
    })
    rows = []
    for result in results:
        key = pair_key(result["left_word"], result["right_word"])
        enriched = enrich_result(result, lp, words, freqs, tokens, f_values, t_values, word_color_scores)
        enriched["strategies"] = sorted(metadata[key]["strategies"])
        enriched["strategy_scores"] = {
            name: float(score)
            for name, score in sorted(metadata[key]["strategy_scores"].items())
        }
        rows.append(enriched)

    report = build_report(rows, examples_per_class=args.examples_per_class)
    report["strategy_summary"] = strategy_summary(rows, candidates)
    report["strategy_overlap"] = strategy_overlap(rows)
    report["run"] = {
        "data_dir": str(Path(args.data_dir).expanduser()),
        "vocab_size": args.vocab_size,
        "pretokenizer": args.pretokenizer,
        "elapsed_seconds": elapsed,
        "lp_bound": float(solution.fun),
        "max_candidates_per_strategy": args.max_candidates_per_strategy,
        "max_total_pairs": args.max_total_pairs,
        "workers": args.workers,
        "batch_size": args.batch_size,
    }
    report["skipped"] = dict(skipped)
    if args.dump_rows:
        report["rows"] = rows

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    LOGGER.info(
        "Alternative pair diagnostics done: checked=%d cuts=%d elapsed=%.3fs output=%s",
        len(rows),
        sum(1 for row in rows if row["has_cut"]),
        elapsed,
        output_path,
    )
    for item in report["strategy_summary"]:
        LOGGER.info(
            "%s proposed=%d tested=%d cuts=%d precision=%.4f avg_violation=%.6g unique_cuts=%d",
            item["strategy"],
            item["proposed"],
            item["tested"],
            item["cuts"],
            item["precision"],
            item["avg_violation"],
            item["unique_cuts"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore alternative word-pair proposal heuristics for pair hull cuts.")
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
    parser.add_argument("--max-colors", type=int, default=96)
    parser.add_argument("--max-pair-rows", type=int, default=250000)
    parser.add_argument("--top-words-per-color", type=int, default=36)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cut-tolerance", type=float, default=1e-6)
    parser.add_argument("--max-candidates-per-strategy", type=int, default=1200)
    parser.add_argument("--max-total-pairs", type=int, default=5000)
    parser.add_argument("--neighbors-per-word", type=int, default=12)
    parser.add_argument("--top-colors-per-word", type=int, default=10)
    parser.add_argument("--examples-per-class", type=int, default=20)
    parser.add_argument("--dump-rows", action="store_true")
    parser.add_argument("--output", default="/tmp/tokenizer_lp_alternative_pair_diagnostics.json")
    return parser.parse_args()


class WordProfile:
    def __init__(self, *, word_idx, word, freq, length, color_scores, color_positions, total_score, top_color_count):
        self.word_idx = word_idx
        self.word = word
        self.normalized_word = normalize_word(word)
        self.freq = float(freq)
        self.length = int(length)
        self.color_scores = color_scores
        self.color_positions = color_positions
        self.total_score = float(total_score)
        self.weighted_score = self.total_score * math.sqrt(max(1.0, self.freq))
        self.norm = math.sqrt(sum(value * value for value in color_scores.values()))
        self.top_colors = frozenset(
            token_idx
            for token_idx, _ in sorted(color_scores.items(), key=lambda item: item[1], reverse=True)[:top_color_count]
        )
        self.ngrams = surface_ngrams(self.normalized_word)


def build_word_profiles(lp, words, freqs, f_values, t_values, args):
    ranked_words = rank_short_fractional_words(
        lp,
        f_values,
        t_values,
        max_words=args.max_words,
        max_word_length=args.max_word_length,
        tolerance=args.cut_tolerance,
    )
    profiles = {}
    for word_idx in ranked_words:
        color_scores = defaultdict(float)
        color_positions = defaultdict(list)
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            edge_value = float(f_values[edge_idx])
            info = lp["nonfree_edge_info"][edge_idx]
            token_idx = info["token_index"]
            token_value = float(t_values[token_idx])
            if args.cut_tolerance < edge_value < 1.0 - args.cut_tolerance and args.cut_tolerance < token_value < 1.0 - args.cut_tolerance:
                length = max(1, int(info["end"] - info["start"]))
                color_scores[token_idx] += min(edge_value, 1.0 - edge_value) * length
                color_positions[token_idx].append((int(info["start"]), int(info["end"])))
        if not color_scores:
            continue
        profiles[word_idx] = WordProfile(
            word_idx=word_idx,
            word=words[word_idx],
            freq=freqs[word_idx],
            length=lp["word_lengths"][word_idx],
            color_scores=dict(color_scores),
            color_positions={key: tuple(value) for key, value in color_positions.items()},
            total_score=sum(color_scores.values()),
            top_color_count=args.top_colors_per_word,
        )
    return profiles


def propose_candidates(lp, f_values, t_values, profiles, args):
    return {
        "baseline_shared_color": baseline_shared_color(lp, f_values, t_values, args),
        "freq_weighted_shared_color": freq_weighted_shared_color(t_values, profiles, args),
        "vector_cosine": vector_neighbors(profiles, args, mode="cosine"),
        "top_color_jaccard": vector_neighbors(profiles, args, mode="jaccard"),
        "surface_ngram": surface_neighbors(profiles, args, mode="ngram"),
        "surface_affix": surface_neighbors(profiles, args, mode="affix"),
    }


def baseline_shared_color(lp, f_values, t_values, args):
    rows, _ = short_word_pair_candidates(
        lp,
        f_values,
        t_values,
        max_words=args.max_words,
        max_word_length=args.max_word_length,
        top_words_per_color=args.top_words_per_color,
        tolerance=args.cut_tolerance,
    )
    return {
        pair_key(left, right): float(score)
        for score, left, right in rows[: args.max_candidates_per_strategy]
    }


def freq_weighted_shared_color(t_values, profiles, args):
    color_rows = defaultdict(list)
    for profile in profiles.values():
        for token_idx, score in profile.color_scores.items():
            token_value = float(t_values[token_idx])
            if args.cut_tolerance < token_value < 1.0 - args.cut_tolerance:
                color_rows[token_idx].append((score * math.sqrt(max(1.0, profile.freq)), profile.word_idx))
    for rows in color_rows.values():
        rows.sort(reverse=True)

    pair_scores = defaultdict(float)
    for token_idx, rows in color_rows.items():
        token_value = min(float(t_values[token_idx]), 1.0 - float(t_values[token_idx]))
        for (left_score, left), (right_score, right) in combinations(rows[: args.top_words_per_color], 2):
            pair_scores[pair_key(left, right)] += min(left_score, right_score) * token_value
    return top_pairs(pair_scores, args.max_candidates_per_strategy)


def vector_neighbors(profiles, args, *, mode):
    word_indices = list(profiles)
    pair_scores = defaultdict(float)
    for left in word_indices:
        left_profile = profiles[left]
        scores = []
        for right in word_indices:
            if left >= right:
                continue
            right_profile = profiles[right]
            if mode == "cosine":
                score = cosine_score(left_profile, right_profile)
            elif mode == "jaccard":
                score = jaccard_score(left_profile.top_colors, right_profile.top_colors)
            else:
                raise ValueError(mode)
            if score <= 0.0:
                continue
            impact = math.sqrt(left_profile.weighted_score * right_profile.weighted_score)
            scores.append((score * impact, right))
        for score, right in sorted(scores, reverse=True)[: args.neighbors_per_word]:
            pair_scores[pair_key(left, right)] = max(pair_scores[pair_key(left, right)], score)
    return top_pairs(pair_scores, args.max_candidates_per_strategy)


def surface_neighbors(profiles, args, *, mode):
    profile_list = list(profiles.values())
    pair_scores = defaultdict(float)
    for left_idx, left in enumerate(profile_list):
        scores = []
        for right in profile_list[left_idx + 1 :]:
            if mode == "ngram":
                similarity = jaccard_score(left.ngrams, right.ngrams)
            elif mode == "affix":
                similarity = affix_similarity(left.normalized_word, right.normalized_word)
            else:
                raise ValueError(mode)
            if similarity <= 0.0:
                continue
            shared_color_bonus = 1.0 + jaccard_score(left.top_colors, right.top_colors)
            impact = math.sqrt(left.weighted_score * right.weighted_score)
            scores.append((similarity * shared_color_bonus * impact, right.word_idx))
        for score, right in sorted(scores, reverse=True)[: args.neighbors_per_word]:
            pair_scores[pair_key(left.word_idx, right)] = max(pair_scores[pair_key(left.word_idx, right)], score)
    return top_pairs(pair_scores, args.max_candidates_per_strategy)


def cosine_score(left: WordProfile, right: WordProfile) -> float:
    if left.norm <= 0.0 or right.norm <= 0.0:
        return 0.0
    if len(left.color_scores) > len(right.color_scores):
        left, right = right, left
    dot = sum(value * right.color_scores.get(token_idx, 0.0) for token_idx, value in left.color_scores.items())
    return dot / (left.norm * right.norm + 1e-12)


def jaccard_score(left, right) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def affix_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    prefix = common_prefix_len(left, right)
    suffix = common_suffix_len(left, right)
    containment = min(len(left), len(right)) if left in right or right in left else 0
    return max(prefix, suffix, containment) / max(len(left), len(right))


def common_prefix_len(left: str, right: str) -> int:
    count = 0
    for l_char, r_char in zip(left, right):
        if l_char != r_char:
            break
        count += 1
    return count


def common_suffix_len(left: str, right: str) -> int:
    return common_prefix_len(left[::-1], right[::-1])


def surface_ngrams(word: str):
    if len(word) <= 3:
        return {word}
    grams = set()
    for n in (2, 3, 4):
        if len(word) >= n:
            grams.update(word[i : i + n] for i in range(len(word) - n + 1))
    return grams


def normalize_word(word: str) -> str:
    return word.lstrip("Ġ").lower()


def top_pairs(pair_scores, limit):
    return dict(sorted(pair_scores.items(), key=lambda item: item[1], reverse=True)[:limit])


def merge_candidates(candidates_by_strategy, max_total_pairs):
    metadata = {}
    for strategy, pair_scores in candidates_by_strategy.items():
        for key, score in pair_scores.items():
            record = metadata.setdefault(key, {"strategies": set(), "strategy_scores": {}, "score": 0.0})
            record["strategies"].add(strategy)
            record["strategy_scores"][strategy] = float(score)
            record["score"] = max(record["score"], normalize_strategy_score(strategy, score))
    ranked = sorted(metadata.items(), key=lambda item: item[1]["score"], reverse=True)[:max_total_pairs]
    return [(data["score"], left, right) for (left, right), data in ranked], {key: data for key, data in ranked}


def normalize_strategy_score(strategy, score):
    if strategy.startswith("surface"):
        return float(score) / 10.0
    if strategy == "freq_weighted_shared_color":
        return float(score) / 10.0
    return float(score)


def pair_key(left, right):
    return tuple(sorted((int(left), int(right))))


def strategy_summary(rows, candidates_by_strategy):
    cut_keys = {pair_key(row["left_word"], row["right_word"]) for row in rows if row["has_cut"]}
    output = []
    for strategy, proposed in sorted(candidates_by_strategy.items()):
        proposed_keys = set(proposed)
        tested = [
            row
            for row in rows
            if strategy in row["strategies"]
        ]
        cuts = [row for row in tested if row["has_cut"]]
        unique_cuts = [row for row in cuts if len(row["strategies"]) == 1]
        output.append(
            {
                "strategy": strategy,
                "proposed": len(proposed_keys),
                "tested": len(tested),
                "cuts": len(cuts),
                "precision": len(cuts) / max(1, len(tested)),
                "avg_violation": float(np.mean([row["violation"] for row in cuts])) if cuts else 0.0,
                "max_violation": max((row["violation"] for row in cuts), default=0.0),
                "unique_cuts": len(unique_cuts),
                "missed_union_cuts": len(cut_keys - {pair_key(row["left_word"], row["right_word"]) for row in cuts}),
            }
        )
    return output


def strategy_overlap(rows):
    strategies = sorted({strategy for row in rows for strategy in row["strategies"]})
    cut_sets = {
        strategy: {
            pair_key(row["left_word"], row["right_word"])
            for row in rows
            if row["has_cut"] and strategy in row["strategies"]
        }
        for strategy in strategies
    }
    return [
        {
            "left": left,
            "right": right,
            "shared_cuts": len(cut_sets[left] & cut_sets[right]),
            "left_cuts": len(cut_sets[left]),
            "right_cuts": len(cut_sets[right]),
        }
        for left, right in combinations(strategies, 2)
    ]


if __name__ == "__main__":
    main()
