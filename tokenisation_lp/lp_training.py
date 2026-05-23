from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from tokenisation_lp.dp_tokenizer import LpDpTokenizer
from tokenisation_lp.lp_tokenizer.datastructures import possibleToken
from tokenisation_lp.lp_tokenizer import helper_functions as hf
from tokenisation_lp.pretokenization import (
    DEFAULT_SPECIAL_TOKENS,
    DEFAULT_UNK_TOKEN,
    build_pretokenizer,
    byte_level_alphabet,
    pretokenize_text,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LpTrainingResult:
    tokenizer: LpDpTokenizer
    vocab: list[str]
    selected_tokens: list[possibleToken]
    objective_value: float
    corpus_token_count_lower_bound: float
    solve_seconds: float
    status: str
    iterations: list["LpSolveIteration"]
    output_path: Path | None = None


@dataclass(frozen=True)
class LpSolveIteration:
    iteration: int
    objective_value: float
    token_count_lower_bound: float
    solve_seconds: float
    status: str
    total_cuts: int
    added_cuts: int
    max_violation: float
    candidates: list[possibleToken]


def train_lp_tokenizer(
    texts: list[str],
    vocab_size: int,
    *,
    pretokenizer_mode: str = "bytelevel",
    special_tokens: list[str] | None = None,
    unk_token: str = DEFAULT_UNK_TOKEN,
    min_token_count: int = 2,
    max_token_length: int | None = 16,
    output_dir: str | Path | None = None,
    cut_rounds: int = 0,
    cuts_per_round: int = 100,
    cut_tolerance: float = 1e-6,
    iteration_callback: Callable[[LpSolveIteration, LpDpTokenizer], None] | None = None,
) -> LpTrainingResult:
    special_tokens = list(special_tokens or DEFAULT_SPECIAL_TOKENS)
    alphabet = byte_level_alphabet()
    multi_token_budget = vocab_size - len(special_tokens) - len(alphabet)
    if multi_token_budget < 0:
        raise ValueError(
            f"vocab_size={vocab_size} is too small for {len(special_tokens)} special "
            f"tokens plus the {len(alphabet)} byte-level alphabet."
        )

    pretokenizer, _ = build_pretokenizer(pretokenizer_mode)
    word_counts = count_pretokenized_strings(texts, pretokenizer)
    LOGGER.info("LP training corpus has %d unique pretokenized strings", len(word_counts))

    def handle_iteration(iteration: LpSolveIteration) -> None:
        if iteration_callback is None:
            return
        selected_for_iteration = round_lp_tokens(
            iteration.candidates,
            multi_token_budget,
            excluded=set(base_vocab),
        )
        iteration_vocab = [*base_vocab, *(token.token for token in selected_for_iteration)]
        iteration_tokenizer = LpDpTokenizer(
            iteration_vocab,
            pretokenizer_mode=pretokenizer_mode,
            unk_token=unk_token,
        )
        iteration_callback(iteration, iteration_tokenizer)

    base_vocab = dedupe_preserve_order([*special_tokens, *alphabet])
    candidates = solve_lp_vocabulary(
        word_counts,
        num_allowed_tokens=multi_token_budget,
        min_token_count=min_token_count,
        max_token_length=max_token_length,
        cut_rounds=cut_rounds,
        cuts_per_round=cuts_per_round,
        cut_tolerance=cut_tolerance,
        iteration_callback=handle_iteration,
    )
    selected_candidates = round_lp_tokens(candidates, multi_token_budget, excluded=set(base_vocab))
    vocab = [*base_vocab, *(token.token for token in selected_candidates)]
    tokenizer = LpDpTokenizer(vocab, pretokenizer_mode=pretokenizer_mode, unk_token=unk_token)

    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir) / "lp_dp_tokenizer.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(output_path)
        LOGGER.info("Saved LP tokenizer to %s", output_path)

    return LpTrainingResult(
        tokenizer=tokenizer,
        vocab=vocab,
        selected_tokens=selected_candidates,
        objective_value=getattr(candidates, "objective_value", float("nan")),
        corpus_token_count_lower_bound=getattr(candidates, "token_count_lower_bound", float("nan")),
        solve_seconds=getattr(candidates, "solve_seconds", float("nan")),
        status=getattr(candidates, "status", "unknown"),
        iterations=getattr(candidates, "iterations", []),
        output_path=output_path,
    )


def count_pretokenized_strings(texts, pretokenizer) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(pretokenize_text(text, pretokenizer))
    return counts


class CandidateList(list):
    objective_value: float
    token_count_lower_bound: float
    solve_seconds: float
    status: str
    iterations: list[LpSolveIteration]


def solve_lp_vocabulary(
    word_counts: Counter[str],
    *,
    num_allowed_tokens: int,
    min_token_count: int = 2,
    max_token_length: int | None = 16,
    cut_rounds: int = 0,
    cuts_per_round: int = 100,
    cut_tolerance: float = 1e-6,
    iteration_callback: Callable[[LpSolveIteration], None] | None = None,
) -> CandidateList:
    if num_allowed_tokens <= 0:
        result = CandidateList()
        result.objective_value = 0.0
        result.token_count_lower_bound = float(sum(len(word) * count for word, count in word_counts.items()))
        result.solve_seconds = 0.0
        result.status = "empty-budget"
        result.iterations = []
        return result

    words = list(word_counts)
    freqs = [word_counts[word] for word in words]
    edges_list, free_edges_list, num_vertices, tokens = prepare_lp_data(
        words,
        freqs,
        min_token_count=min_token_count,
        max_token_length=max_token_length,
    )
    if not tokens:
        result = CandidateList()
        result.objective_value = 0.0
        result.token_count_lower_bound = float(sum(len(word) * count for word, count in word_counts.items()))
        result.solve_seconds = 0.0
        result.status = "no-candidates"
        result.iterations = []
        return result

    lp = build_standard_form(edges_list, freqs, tokens, free_edges_list, num_vertices)
    LOGGER.info(
        "Solving LP with %d candidate tokens, %d non-free edges, %d free edges",
        lp["num_tokens"],
        lp["num_nonfree_edges"],
        lp["num_free_edges"],
    )
    a_ub = lp["A_ub"]
    b_ub = lp["b_ub"].copy()
    b_ub[lp["budget_row"]] = float(num_allowed_tokens)
    existing_cut_keys: set[tuple[int, int, int]] = set()
    iterations: list[LpSolveIteration] = []
    final_candidates = CandidateList()

    for iteration in range(cut_rounds + 1):
        start = time.monotonic()
        solution = linprog(
            c=lp["c"],
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=lp["A_eq"],
            b_eq=lp["b_eq"],
            bounds=list(zip(lp["lb"], lp["ub"])),
            method="highs",
        )
        solve_seconds = time.monotonic() - start
        if not solution.success:
            raise RuntimeError(f"SciPy HiGHS LP solve failed: {solution.message}")

        candidates = candidates_from_solution(tokens, lp, solution.x)
        candidates.objective_value = float(solution.fun)
        candidates.token_count_lower_bound = float(solution.fun)
        candidates.solve_seconds = solve_seconds
        candidates.status = str(solution.message)

        added_cuts = 0
        max_violation = 0.0
        cut_matrix = None
        cut_keys: list[tuple[int, int, int]] = []
        if iteration < cut_rounds:
            cut_matrix, cut_keys, max_violation = separate_byte_boundary_cuts(
                lp,
                solution.x,
                existing_cut_keys=existing_cut_keys,
                max_cuts=cuts_per_round,
                tolerance=cut_tolerance,
            )
            added_cuts = len(cut_keys)

        iteration_result = LpSolveIteration(
            iteration=iteration,
            objective_value=float(solution.fun),
            token_count_lower_bound=float(solution.fun),
            solve_seconds=solve_seconds,
            status=str(solution.message),
            total_cuts=len(existing_cut_keys),
            added_cuts=added_cuts,
            max_violation=max_violation,
            candidates=list(candidates),
        )
        iterations.append(iteration_result)
        LOGGER.info(
            "LP iteration %d solved in %.3fs: objective=%.3f nonzero_tokens=%d "
            "active_cuts=%d next_cuts=%d max_cut_violation=%.6g",
            iteration,
            solve_seconds,
            solution.fun,
            len(candidates),
            len(existing_cut_keys),
            added_cuts,
            max_violation,
        )
        if iteration_callback is not None:
            iteration_callback(iteration_result)

        final_candidates = candidates
        if added_cuts == 0:
            break

        existing_cut_keys.update(cut_keys)
        a_ub = sp.vstack([a_ub, cut_matrix], format="csr")
        b_ub = np.concatenate([b_ub, np.zeros(added_cuts, dtype=float)])

    final_candidates.iterations = iterations
    return final_candidates


def candidates_from_solution(tokens, lp, x_values) -> CandidateList:
    t_values = x_values[lp["num_nonfree_edges"] + lp["num_free_edges"] :]
    candidates = CandidateList(
        possibleToken(
            token=tokens[i].token,
            lp_value=float(t_values[i]),
            instance_count=tokens[i].token_instance_count,
            index=i,
        )
        for i in range(len(tokens))
        if t_values[i] > 1e-9
    )
    candidates.sort(key=lambda token: (token.lp_value, token.token_instance_count, len(token.token)), reverse=True)
    return candidates


def separate_byte_boundary_cuts(
    lp,
    x_values,
    *,
    existing_cut_keys: set[tuple[int, int, int]],
    max_cuts: int,
    tolerance: float,
):
    if max_cuts <= 0:
        return None, [], 0.0

    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    f_values = x_values[:num_f]
    t_values = x_values[num_f + num_g :]

    violations = []
    for key, edge_indices in lp["boundary_crossings"].items():
        if key in existing_cut_keys:
            continue
        token_index = key[2]
        lhs_value = float(f_values[edge_indices].sum())
        violation = lhs_value - float(t_values[token_index])
        if violation > tolerance:
            violations.append((violation, key, edge_indices))

    if not violations:
        return None, [], 0.0

    violations.sort(key=lambda item: item[0], reverse=True)
    selected = violations[:max_cuts]
    rows = []
    cols = []
    data = []
    num_vars = len(x_values)

    for row_idx, (_, key, edge_indices) in enumerate(selected):
        rows.extend([row_idx] * (len(edge_indices) + 1))
        cols.extend([*edge_indices, num_f + num_g + key[2]])
        data.extend([1.0] * len(edge_indices))
        data.append(-1.0)

    cut_matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(selected), num_vars),
        dtype=float,
    ).tocsr()
    return cut_matrix, [key for _, key, _ in selected], float(selected[0][0])


def prepare_lp_data(
    words: list[str],
    freqs: list[int],
    *,
    min_token_count: int,
    max_token_length: int | None,
):
    edges_list = []
    free_edges_list = []
    num_vertices = []
    token_lists = []

    for word in words:
        limit = len(word) if max_token_length is None else min(len(word), max_token_length)
        edges = hf.get_all_nonFree_substrings_upto_len_t(word, limit)
        token_list = hf.get_tokens_upto_len_t(word, limit)
        edges_list.append(edges)
        token_lists.append(token_list)
        free_edges_list.append(hf.get_all_free_substrings(word))
        num_vertices.append(len(word) + 1)

    tokens = list(set(token for row in token_lists for token in row))
    hf.update_token_instance_counts(tokens, freqs, edges_list)
    tokens_to_keep = [token for token in tokens if token.token_instance_count >= min_token_count]
    keep_set = {token.token for token in tokens_to_keep}
    filtered_edges = [[edge for edge in row if edge.token in keep_set] for row in edges_list]
    return filtered_edges, free_edges_list, num_vertices, tokens_to_keep


def build_standard_form(edges_list, edge_weights, tokens, free_edges_list, num_vertices_list):
    token_index = {token.token: i for i, token in enumerate(tokens)}
    a_rows, a_cols, a_data = [], [], []
    b_rows, b_cols, b_data = [], [], []
    m_rows, m_cols, m_data = [], [], []
    rhs_parts = []
    nonfree_cost_parts = []
    free_cost_parts = []
    boundary_crossings = defaultdict(list)
    vertex_offset = edge_offset = free_edge_offset = 0

    for word_idx, (edges, free_edges, weight, num_vertices) in enumerate(
        zip(edges_list, free_edges_list, edge_weights, num_vertices_list)
    ):
        for local_idx, edge in enumerate(edges):
            edge_idx = edge_offset + local_idx
            edge_token_index = token_index[edge.token]
            a_rows.extend([vertex_offset + edge.start, vertex_offset + edge.end])
            a_cols.extend([edge_idx, edge_idx])
            a_data.extend([1.0, -1.0])
            m_rows.append(edge_idx)
            m_cols.append(edge_token_index)
            m_data.append(1.0)
            for boundary in range(edge.start, edge.end):
                boundary_crossings[(word_idx, boundary, edge_token_index)].append(edge_idx)

        for local_idx, edge in enumerate(free_edges):
            b_rows.extend([vertex_offset + edge.start, vertex_offset + edge.end])
            b_cols.extend([free_edge_offset + local_idx, free_edge_offset + local_idx])
            b_data.extend([1.0, -1.0])

        rhs = np.zeros(num_vertices, dtype=float)
        rhs[0] = 1.0
        rhs[-1] = -1.0
        rhs_parts.append(rhs)
        nonfree_cost_parts.append(np.full(len(edges), float(weight), dtype=float))
        free_cost_parts.append(np.full(len(free_edges), float(weight), dtype=float))

        vertex_offset += num_vertices
        edge_offset += len(edges)
        free_edge_offset += len(free_edges)

    num_nonfree = edge_offset
    num_free = free_edge_offset
    num_tokens = len(tokens)
    num_vars = num_nonfree + num_free + num_tokens

    a_constraint = sp.coo_matrix(
        (a_data, (a_rows, a_cols)), shape=(vertex_offset, num_nonfree), dtype=float
    ).tocsr()
    b_constraint = sp.coo_matrix(
        (b_data, (b_rows, b_cols)), shape=(vertex_offset, num_free), dtype=float
    ).tocsr()
    m_constraint = sp.coo_matrix(
        (m_data, (m_rows, m_cols)), shape=(num_nonfree, num_tokens), dtype=float
    ).tocsr()

    a_eq = sp.hstack(
        [a_constraint, b_constraint, sp.csr_matrix((vertex_offset, num_tokens), dtype=float)],
        format="csr",
    )
    a_ub_flow = sp.hstack(
        [
            sp.identity(num_nonfree, format="csr", dtype=float),
            sp.csr_matrix((num_nonfree, num_free), dtype=float),
            -m_constraint,
        ],
        format="csr",
    )
    budget_cols = np.arange(num_nonfree + num_free, num_vars)
    budget = sp.coo_matrix(
        (np.ones(num_tokens, dtype=float), (np.zeros(num_tokens, dtype=int), budget_cols)),
        shape=(1, num_vars),
        dtype=float,
    ).tocsr()

    return {
        "A_eq": a_eq,
        "b_eq": np.concatenate(rhs_parts),
        "A_ub": sp.vstack([a_ub_flow, budget], format="csr"),
        "b_ub": np.zeros(num_nonfree + 1, dtype=float),
        "c": np.concatenate(
            [
                np.concatenate(nonfree_cost_parts) if nonfree_cost_parts else np.array([], dtype=float),
                np.concatenate(free_cost_parts) if free_cost_parts else np.array([], dtype=float),
                np.zeros(num_tokens, dtype=float),
            ]
        ),
        "lb": np.zeros(num_vars, dtype=float),
        "ub": np.ones(num_vars, dtype=float),
        "num_nonfree_edges": num_nonfree,
        "num_free_edges": num_free,
        "num_tokens": num_tokens,
        "budget_row": num_nonfree,
        "boundary_crossings": dict(boundary_crossings),
    }


def round_lp_tokens(
    candidates: list[possibleToken],
    budget: int,
    *,
    excluded: set[str] | None = None,
) -> list[possibleToken]:
    excluded = set(excluded or ())
    selected = []
    for token in candidates:
        if token.token in excluded:
            continue
        selected.append(token)
        excluded.add(token.token)
        if len(selected) == budget:
            break
    return selected


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
