from __future__ import annotations

import argparse
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    HighsWarmLpSolver,
    all_word_token_colors,
    build_standard_form,
    count_pretokenized_strings,
    enumerate_word_edge_paths_by_pattern,
    prepare_lp_data,
    rank_short_fractional_words,
    separate_short_word_full_hull_cut_specs,
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
    LOGGER.info("Base LP bound %.3f", solution.fun)
    if args.apply_individual_hulls:
        solution = apply_individual_hulls(args, lp, solver, solution)

    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    f_values = solution.x[:num_f]
    g_values = solution.x[num_f : num_f + num_g]
    t_values = solution.x[num_f + num_g :]
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

    candidates = []
    if args.mode in {"color", "both"}:
        candidates.extend(color_centered_subsets(context, args))
    if args.mode in {"cluster", "both"}:
        candidates.extend(clustered_word_subsets(context, args))

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = tuple(sorted(candidate))
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(key)
    LOGGER.info("Testing %d unique subset candidates", len(unique_candidates))

    start = time.monotonic()
    cuts = []
    skipped = defaultdict(int)
    pair_cut_cache = {}
    for subset in unique_candidates[: args.max_subsets]:
        if args.skip_subsets_with_pair_cuts and len(subset) > 2:
            pair_cut_found = False
            for pair in combinations(subset, 2):
                pair_key = tuple(sorted(pair))
                if pair_key not in pair_cut_cache:
                    pair_cut_cache[pair_key] = context.separate_subset(pair_key)
                if isinstance(pair_cut_cache[pair_key], SubsetCut):
                    pair_cut_found = True
                    break
            if pair_cut_found:
                skipped["pair_cut"] += 1
                continue
        result = context.separate_subset(subset)
        if result is None:
            skipped["no_cut"] += 1
            continue
        if isinstance(result, str):
            skipped[result] += 1
            continue
        cuts.append(result)
    cuts.sort(key=lambda item: item.violation, reverse=True)
    LOGGER.info(
        "Subset scan done: tested=%d cuts=%d elapsed=%.3fs skipped=%s",
        min(len(unique_candidates), args.max_subsets),
        len(cuts),
        time.monotonic() - start,
        dict(skipped),
    )
    for cut in cuts[: args.print_cuts]:
        word_desc = " / ".join(
            f"{context.words[word_idx]!r}(len={context.lp['word_lengths'][word_idx]},freq={context.freqs[word_idx]})"
            for word_idx in cut.word_indices
        )
        LOGGER.info(
            "cut violation=%.6g words=%d colors=%d rows=%d build=%.3fs solve=%.3fs %s",
            cut.violation,
            len(cut.word_indices),
            cut.num_colors,
            cut.num_rows,
            cut.build_seconds,
            cut.solve_seconds,
            word_desc,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore arbitrary small word-subset upward-hull cuts.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8)
    parser.add_argument("--lp-solution-cache-dir", default="/tmp/tokenizer_lp_solution_cache")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--mode", default="both", choices=("color", "cluster", "both"))
    parser.add_argument("--apply-individual-hulls", action="store_true")
    parser.add_argument("--individual-rounds", type=int, default=3)
    parser.add_argument("--individual-max-words", type=int, default=12000)
    parser.add_argument("--individual-max-length", type=int, default=12)
    parser.add_argument("--individual-max-colors", type=int, default=96)
    parser.add_argument("--max-paths", type=int, default=100000)
    parser.add_argument("--max-colors", type=int, default=128)
    parser.add_argument("--max-product-rows", type=int, default=200000)
    parser.add_argument("--cut-tolerance", type=float, default=1e-6)
    parser.add_argument("--candidate-words", type=int, default=600)
    parser.add_argument("--subset-size", type=int, default=3)
    parser.add_argument("--max-subsets", type=int, default=400)
    parser.add_argument("--print-cuts", type=int, default=20)
    parser.add_argument("--top-colors", type=int, default=80)
    parser.add_argument("--top-words-per-color", type=int, default=8)
    parser.add_argument("--cluster-neighbors", type=int, default=8)
    parser.add_argument("--top-colors-per-word", type=int, default=10)
    parser.add_argument(
        "--skip-subsets-with-pair-cuts",
        action="store_true",
        help="For subset size >2, skip subsets whose contained pair already has a hull cut.",
    )
    return parser.parse_args()


def apply_individual_hulls(args, lp, solver, solution):
    existing = set()
    current = solution
    num_vars = len(lp["c"])
    for round_idx in range(args.individual_rounds):
        num_f = lp["num_nonfree_edges"]
        num_g = lp["num_free_edges"]
        f_values = current.x[:num_f]
        g_values = current.x[num_f : num_f + num_g]
        t_values = current.x[num_f + num_g :]
        start = time.monotonic()
        violations = separate_short_word_full_hull_cut_specs(
            lp,
            f_values,
            g_values,
            t_values,
            existing_cut_keys=existing,
            tolerance=args.cut_tolerance,
            max_words=args.individual_max_words,
            max_word_length=args.individual_max_length,
            max_colors=args.individual_max_colors,
            max_paths=args.max_paths,
        )
        LOGGER.info(
            "individual hull round %d cuts=%d max=%.6g sep=%.3fs",
            round_idx,
            len(violations),
            max((cut[0] for cut in violations), default=0.0),
            time.monotonic() - start,
        )
        if not violations:
            break
        matrix, rhs = cut_specs_to_matrix(violations, num_vars)
        existing.update(cut[1] for cut in violations)
        solver.add_ub_rows(matrix, rhs)
        current = solver.solve()
        LOGGER.info("individual hull round %d bound %.3f", round_idx, current.fun)
    return current


def cut_specs_to_matrix(violations, num_vars: int):
    rows = []
    cols = []
    data = []
    rhs_values = []
    for row_idx, (_, _, entries, rhs) in enumerate(violations):
        rhs_values.append(float(rhs))
        for col_idx, coefficient in entries:
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(float(coefficient))
    return sp.coo_matrix((data, (rows, cols)), shape=(len(violations), num_vars), dtype=float).tocsr(), np.array(rhs_values)


@dataclass(frozen=True)
class SubsetCut:
    violation: float
    word_indices: tuple[int, ...]
    num_colors: int
    num_rows: int
    build_seconds: float
    solve_seconds: float


class SubsetHullContext:
    def __init__(
        self,
        *,
        lp,
        words,
        freqs,
        f_values,
        g_values,
        t_values,
        max_paths,
        max_colors,
        max_product_rows,
        tolerance,
    ):
        self.lp = lp
        self.words = words
        self.freqs = freqs
        self.f_values = f_values
        self.g_values = g_values
        self.t_values = t_values
        self.max_paths = max_paths
        self.max_colors = max_colors
        self.max_product_rows = max_product_rows
        self.tolerance = tolerance
        self.path_cache = {}
        self.paths_by_word = {}
        self.colors_by_word = {}
        self.vector_by_word = {}

    def word_paths(self, word_idx: int):
        if word_idx not in self.paths_by_word:
            self.paths_by_word[word_idx] = enumerate_word_edge_paths_by_pattern(
                self.lp,
                word_idx,
                max_paths=self.max_paths,
                cache=self.path_cache,
            )
        return self.paths_by_word[word_idx]

    def word_colors(self, word_idx: int):
        if word_idx not in self.colors_by_word:
            self.colors_by_word[word_idx] = all_word_token_colors(self.lp, word_idx)
        return self.colors_by_word[word_idx]

    def fractional_vector(self, word_idx: int):
        if word_idx in self.vector_by_word:
            return self.vector_by_word[word_idx]
        values = defaultdict(float)
        for edge_idx in self.lp["word_nonfree_edges"].get(word_idx, []):
            edge_value = float(self.f_values[edge_idx])
            if edge_value <= self.tolerance:
                continue
            info = self.lp["nonfree_edge_info"][edge_idx]
            token_idx = info["token_index"]
            token_value = float(self.t_values[token_idx])
            if self.tolerance < token_value < 1.0 - self.tolerance:
                values[token_idx] += edge_value
        self.vector_by_word[word_idx] = dict(values)
        return self.vector_by_word[word_idx]

    def separate_subset(self, word_indices):
        selected_tokens = tuple(sorted(set().union(*(self.word_colors(word_idx) for word_idx in word_indices))))
        if len(selected_tokens) < 2:
            return "few_colors"
        if len(selected_tokens) > self.max_colors:
            return "too_many_colors"

        paths = []
        product_rows = 1
        for word_idx in word_indices:
            word_paths = self.word_paths(word_idx)
            if word_paths is None or len(word_paths) < 2:
                return "bad_paths"
            paths.append(word_paths)
            product_rows *= len(word_paths)
            if product_rows > self.max_product_rows:
                return "too_many_rows"
        return separate_upward_hull_subset(
            self,
            tuple(word_indices),
            selected_tokens,
            paths,
            product_rows,
        )


def color_centered_subsets(context: SubsetHullContext, args):
    ranked_words = rank_short_fractional_words(
        context.lp,
        context.f_values,
        context.t_values,
        max_words=args.candidate_words,
        max_word_length=args.individual_max_length,
        tolerance=args.cut_tolerance,
    )
    color_rows = defaultdict(list)
    for word_idx in ranked_words:
        for token_idx, value in context.fractional_vector(word_idx).items():
            token_value = float(context.t_values[token_idx])
            score = value * min(token_value, 1.0 - token_value) * float(context.freqs[word_idx])
            if score > 0:
                color_rows[token_idx].append((score, word_idx))
    color_scores = [
        (min(float(context.t_values[token_idx]), 1.0 - float(context.t_values[token_idx])) * sum(score for score, _ in rows), token_idx)
        for token_idx, rows in color_rows.items()
        if args.cut_tolerance < float(context.t_values[token_idx]) < 1.0 - args.cut_tolerance
    ]
    color_scores.sort(reverse=True)

    subsets = []
    for _, token_idx in color_scores[: args.top_colors]:
        rows = sorted(color_rows[token_idx], reverse=True)[: args.top_words_per_color]
        words = [word_idx for _, word_idx in rows]
        for combo in combinations(words, args.subset_size):
            subsets.append(combo)
    LOGGER.info("color-centered proposed %d subsets", len(subsets))
    return subsets


def clustered_word_subsets(context: SubsetHullContext, args):
    ranked_words = rank_short_fractional_words(
        context.lp,
        context.f_values,
        context.t_values,
        max_words=args.candidate_words,
        max_word_length=args.individual_max_length,
        tolerance=args.cut_tolerance,
    )
    top_sets = {}
    norms = {}
    vectors = {}
    for word_idx in ranked_words:
        vector = context.fractional_vector(word_idx)
        if not vector:
            continue
        vectors[word_idx] = vector
        top_sets[word_idx] = {
            token_idx
            for token_idx, _ in sorted(vector.items(), key=lambda item: item[1], reverse=True)[: args.top_colors_per_word]
        }
        norms[word_idx] = math.sqrt(sum(value * value for value in vector.values()))

    subsets = []
    word_list = list(vectors)
    for anchor in word_list:
        scores = []
        anchor_set = top_sets[anchor]
        anchor_vector = vectors[anchor]
        for other in word_list:
            if other == anchor:
                continue
            other_set = top_sets[other]
            union_size = len(anchor_set | other_set)
            if union_size == 0:
                continue
            jaccard = len(anchor_set & other_set) / union_size
            dot = sum(value * vectors[other].get(token_idx, 0.0) for token_idx, value in anchor_vector.items())
            cosine = dot / (norms[anchor] * norms[other] + 1e-12)
            scores.append((0.5 * jaccard + 0.5 * cosine, other))
        scores.sort(reverse=True)
        neighbors = [other for _, other in scores[: args.cluster_neighbors]]
        for combo in combinations(neighbors, args.subset_size - 1):
            subsets.append((anchor, *combo))
    LOGGER.info("cluster proposed %d subsets", len(subsets))
    return subsets


def separate_upward_hull_subset(
    context: SubsetHullContext,
    word_indices,
    selected_tokens,
    paths_by_word,
    product_rows: int,
):
    lp = context.lp
    num_f = lp["num_nonfree_edges"]
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    selected_set = set(selected_tokens)
    word_columns = []
    current_values = []
    for word_idx in word_indices:
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            word_columns.append(edge_idx)
            current_values.append(float(context.f_values[edge_idx]))
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            word_columns.append(num_f + edge_idx)
            current_values.append(float(context.g_values[edge_idx]))

    build_start = time.monotonic()
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
    for path_tuple in product(*paths_by_word):
        required = set()
        for _, path_tokens in path_tuple:
            required.update(path_tokens & selected_set)
        for path_columns, _ in path_tuple:
            for col_idx in path_columns:
                pos = col_position[col_idx]
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

    constraints = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(row_idx + 1, num_vars),
        dtype=float,
    ).tocsr()
    objective = np.zeros(num_vars, dtype=float)
    for pos, value in enumerate(current_values):
        objective[a_pos_offset + pos] = -value
        objective[a_neg_offset + pos] = value
    for pos, token_idx in enumerate(selected_tokens):
        token_value = float(context.t_values[token_idx])
        objective[b_pos_offset + pos] = -token_value
        objective[b_neg_offset + pos] = token_value
    objective[gamma_col] = -1.0
    bounds = [(0.0, None)] * num_vars
    bounds[gamma_col] = (None, None)
    build_seconds = time.monotonic() - build_start

    solve_start = time.monotonic()
    result = linprog(
        c=objective,
        A_ub=constraints,
        b_ub=np.array(rhs, dtype=float),
        bounds=bounds,
        method="highs",
    )
    solve_seconds = time.monotonic() - solve_start
    if not result.success:
        return "solve_failed"
    violation = -float(result.fun)
    if violation <= context.tolerance:
        return None
    return SubsetCut(
        violation=violation,
        word_indices=word_indices,
        num_colors=len(selected_tokens),
        num_rows=product_rows,
        build_seconds=build_seconds,
        solve_seconds=solve_seconds,
    )


if __name__ == "__main__":
    main()
