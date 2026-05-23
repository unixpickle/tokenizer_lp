from __future__ import annotations

import logging
import time
import hashlib
from itertools import combinations
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
    cut_families: tuple[str, ...] = ("boundary",),
    lp_solution_cache_dir: str | Path | None = None,
    lp_solver: str = "highspy",
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
        cut_families=cut_families,
        lp_solution_cache_dir=lp_solution_cache_dir,
        lp_solver=lp_solver,
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
    cut_families: tuple[str, ...] = ("boundary",),
    lp_solution_cache_dir: str | Path | None = None,
    lp_solver: str = "highspy",
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
    highs_solver = None
    if lp_solver == "highspy":
        highs_solver = HighsWarmLpSolver(
            c=lp["c"],
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=lp["A_eq"],
            b_eq=lp["b_eq"],
            lb=lp["lb"],
            ub=lp["ub"],
            cache_dir=lp_solution_cache_dir,
        )
    elif lp_solver != "scipy":
        raise ValueError(f"Unsupported LP solver {lp_solver!r}. Expected 'highspy' or 'scipy'.")

    for iteration in range(cut_rounds + 1):
        start = time.monotonic()
        if highs_solver is not None:
            solution = highs_solver.solve()
        else:
            solution = solve_linprog_cached(
                c=lp["c"],
                A_ub=a_ub,
                b_ub=b_ub,
                A_eq=lp["A_eq"],
                b_eq=lp["b_eq"],
                bounds=list(zip(lp["lb"], lp["ub"])),
                cache_dir=lp_solution_cache_dir,
            )
        solve_seconds = time.monotonic() - start
        if not solution.success:
            raise RuntimeError(f"LP solve failed with {lp_solver}: {solution.message}")

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
            cut_matrix, cut_rhs, cut_keys, max_violation = separate_cuts(
                lp,
                solution.x,
                existing_cut_keys=existing_cut_keys,
                max_cuts=cuts_per_round,
                tolerance=cut_tolerance,
                families=cut_families,
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
        if highs_solver is not None:
            highs_solver.add_ub_rows(cut_matrix, cut_rhs)
        else:
            a_ub = sp.vstack([a_ub, cut_matrix], format="csr")
            b_ub = np.concatenate([b_ub, cut_rhs])

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


def solve_linprog_cached(*, c, A_ub, b_ub, A_eq, b_eq, bounds, cache_dir=None):
    if cache_dir is None:
        return linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    key = lp_cache_key(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    solution_path = cache_path / f"{key}.npz"

    if solution_path.exists():
        cached = np.load(solution_path, allow_pickle=False)
        LOGGER.info("Loaded LP solution cache hit: %s", solution_path)
        return SimpleNamespace(
            success=bool(cached["success"]),
            x=cached["x"],
            fun=float(cached["fun"]),
            message=str(cached["message"]),
        )

    solution = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if solution.success:
        np.savez_compressed(
            solution_path,
            success=np.array(solution.success),
            x=solution.x,
            fun=np.array(solution.fun, dtype=float),
            message=np.array(str(solution.message)),
        )
        LOGGER.info("Saved LP solution cache entry: %s", solution_path)
    return solution


class HighsWarmLpSolver:
    """Thin HiGHS wrapper that keeps one model alive as cut rows are added."""

    def __init__(
        self,
        *,
        c,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
        lb,
        ub,
        cache_dir=None,
    ):
        import highspy

        self.highspy = highspy
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.c = np.asarray(c, dtype=float)
        self.A_ub = A_ub.tocsr()
        self.b_ub = np.asarray(b_ub, dtype=float)
        self.A_eq = A_eq.tocsr()
        self.b_eq = np.asarray(b_eq, dtype=float)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.bounds = list(zip(self.lb, self.ub))

        self.highs = highspy.Highs()
        self.highs.setOptionValue("output_flag", False)
        self.highs.setOptionValue("solver", "simplex")
        self.highs.setOptionValue("simplex_strategy", 1)
        self.highs.setOptionValue("presolve", "off")
        self.highs.passModel(self._build_model())

    def solve(self):
        cached = self._load_cache()
        if cached is not None:
            return cached

        self.highs.run()
        model_status = self.highs.getModelStatus()
        success = model_status == self.highspy.HighsModelStatus.kOptimal
        message = self.highs.modelStatusToString(model_status)
        if not success:
            return SimpleNamespace(success=False, x=None, fun=float("nan"), message=message)

        solution = self.highs.getSolution()
        x_values = np.array(solution.col_value, dtype=float)
        objective = float(self.highs.getObjectiveValue())
        result = SimpleNamespace(
            success=True,
            x=x_values,
            fun=objective,
            message=message,
        )
        self._save_cache(result)
        return result

    def add_ub_rows(self, A_rows, b_rows) -> None:
        if A_rows is None:
            return
        A_rows = A_rows.tocsr()
        b_rows = np.asarray(b_rows, dtype=float)
        if A_rows.shape[1] != self.c.shape[0]:
            raise ValueError(
                f"Cut matrix has {A_rows.shape[1]} columns, expected {self.c.shape[0]}."
            )
        if A_rows.shape[0] != b_rows.shape[0]:
            raise ValueError("Cut matrix row count must match cut RHS length.")
        if A_rows.shape[0] == 0:
            return

        row_lower = np.full(A_rows.shape[0], -self.highspy.kHighsInf, dtype=float)
        row_upper = b_rows.astype(float, copy=False)
        self.highs.addRows(
            A_rows.shape[0],
            row_lower,
            row_upper,
            A_rows.nnz,
            A_rows.indptr.astype(np.int32, copy=False),
            A_rows.indices.astype(np.int32, copy=False),
            A_rows.data.astype(float, copy=False),
        )
        self.A_ub = sp.vstack([self.A_ub, A_rows], format="csr")
        self.b_ub = np.concatenate([self.b_ub, b_rows])

    def _build_model(self):
        highspy = self.highspy
        constraint_matrix = sp.vstack([self.A_eq, self.A_ub], format="csr")
        row_lower = np.concatenate(
            [
                self.b_eq,
                np.full(self.A_ub.shape[0], -highspy.kHighsInf, dtype=float),
            ]
        )
        row_upper = np.concatenate([self.b_eq, self.b_ub])

        matrix = highspy.HighsSparseMatrix()
        matrix.num_col_ = int(constraint_matrix.shape[1])
        matrix.num_row_ = int(constraint_matrix.shape[0])
        matrix.format_ = highspy.MatrixFormat.kRowwise
        matrix.start_ = constraint_matrix.indptr.astype(np.int32, copy=False)
        matrix.p_end_ = np.array([], dtype=np.int32)
        matrix.index_ = constraint_matrix.indices.astype(np.int32, copy=False)
        matrix.value_ = constraint_matrix.data.astype(float, copy=False)

        lp = highspy.HighsLp()
        lp.num_col_ = int(self.c.shape[0])
        lp.num_row_ = int(constraint_matrix.shape[0])
        lp.col_cost_ = self.c
        lp.col_lower_ = self.lb
        lp.col_upper_ = self.ub
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper
        lp.a_matrix_ = matrix
        lp.sense_ = highspy.ObjSense.kMinimize
        lp.offset_ = 0.0
        return lp

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        key = lp_cache_key(
            c=self.c,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            bounds=self.bounds,
        )
        return self.cache_dir / f"{key}.npz"

    def _load_cache(self):
        solution_path = self._cache_path()
        if solution_path is None or not solution_path.exists():
            return None
        cached = np.load(solution_path, allow_pickle=False)
        LOGGER.info("Loaded LP solution cache hit: %s", solution_path)
        return SimpleNamespace(
            success=bool(cached["success"]),
            x=cached["x"],
            fun=float(cached["fun"]),
            message=str(cached["message"]),
        )

    def _save_cache(self, solution) -> None:
        solution_path = self._cache_path()
        if solution_path is None or not solution.success:
            return
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            solution_path,
            success=np.array(solution.success),
            x=solution.x,
            fun=np.array(solution.fun, dtype=float),
            message=np.array(str(solution.message)),
        )
        LOGGER.info("Saved LP solution cache entry: %s", solution_path)


def lp_cache_key(*, c, A_ub, b_ub, A_eq, b_eq, bounds) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"tokenizer-lp-highs-v1")
    hash_array(hasher, np.asarray(c, dtype=float))
    hash_sparse(hasher, A_ub)
    hash_array(hasher, np.asarray(b_ub, dtype=float))
    hash_sparse(hasher, A_eq)
    hash_array(hasher, np.asarray(b_eq, dtype=float))
    hash_array(hasher, np.asarray(bounds, dtype=float))
    return hasher.hexdigest()


def hash_array(hasher, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(str(contiguous.shape).encode("ascii"))
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(contiguous.view(np.uint8))


def hash_sparse(hasher, matrix) -> None:
    csr = matrix.tocsr()
    hasher.update(str(csr.shape).encode("ascii"))
    hash_array(hasher, csr.indptr)
    hash_array(hasher, csr.indices)
    hash_array(hasher, csr.data)


def separate_cuts(
    lp,
    x_values,
    *,
    existing_cut_keys: set[tuple],
    max_cuts: int,
    tolerance: float,
    families: tuple[str, ...],
):
    if max_cuts <= 0:
        return None, np.array([], dtype=float), [], 0.0

    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    f_values = x_values[:num_f]
    t_values = x_values[num_f + num_g :]

    violations = []
    family_set = set(families)
    if "boundary" in family_set:
        violations.extend(
            separate_byte_boundary_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "word_packing" in family_set:
        violations.extend(
            separate_word_packing_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "path_config" in family_set:
        violations.extend(
            separate_path_config_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "path_multicover" in family_set:
        violations.extend(
            separate_path_multicover_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "global_token_packing" in family_set:
        violations.extend(
            separate_global_token_packing_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "global_pair_packing" in family_set:
        violations.extend(
            separate_global_rank_packing_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                rank=2,
                top_colors_per_word=8,
            )
        )
    if "global_triple_packing" in family_set:
        violations.extend(
            separate_global_rank_packing_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                rank=3,
                top_colors_per_word=7,
            )
        )
    if "window_overlap" in family_set:
        violations.extend(
            separate_window_overlap_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "window_overlap_deep" in family_set:
        violations.extend(
            separate_window_overlap_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                max_words=1000,
                max_rank=6,
                window_lengths=(4, 5, 6, 8, 10, 12, 16),
            )
        )
    if "word_path_cover" in family_set:
        violations.extend(
            separate_word_path_cover_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "window_pair" in family_set:
        violations.extend(
            separate_window_pair_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )

    if not violations:
        return None, np.array([], dtype=float), [], 0.0

    violations.sort(key=lambda item: item[0], reverse=True)
    selected = violations[:max_cuts]
    rows = []
    cols = []
    data = []
    num_vars = len(x_values)
    rhs_values = []

    for row_idx, (_, _, entries, rhs) in enumerate(selected):
        rhs_values.append(float(rhs))
        for col_idx, coefficient in entries:
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(float(coefficient))

    cut_matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(selected), num_vars),
        dtype=float,
    ).tocsr()
    return (
        cut_matrix,
        np.array(rhs_values, dtype=float),
        [key for _, key, _, _ in selected],
        float(selected[0][0]),
    )


def separate_byte_boundary_cut_specs(
    lp,
    f_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
):
    violations = []
    for key, edge_indices in lp["boundary_crossings"].items():
        full_key = ("boundary", *key)
        if full_key in existing_cut_keys:
            continue
        token_index = key[2]
        lhs_value = float(f_values[edge_indices].sum())
        violation = lhs_value - float(t_values[token_index])
        if violation > tolerance:
            entries = [(edge_idx, 1.0) for edge_idx in edge_indices]
            entries.append((lp["num_nonfree_edges"] + lp["num_free_edges"] + token_index, -1.0))
            violations.append((violation, full_key, entries, 0.0))
    return violations


def separate_word_packing_cut_specs(
    lp,
    f_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
):
    violations = []
    for key, edge_indices in lp["word_token_occurrences"].items():
        word_idx, token_index = key
        max_pack = lp["word_token_max_pack"][key]
        if max_pack <= 1:
            continue
        full_key = ("word_packing", word_idx, token_index)
        if full_key in existing_cut_keys:
            continue
        lhs_value = float(f_values[edge_indices].sum())
        rhs_value = float(max_pack * t_values[token_index])
        violation = lhs_value - rhs_value
        if violation > tolerance:
            entries = [(edge_idx, 1.0) for edge_idx in edge_indices]
            entries.append((lp["num_nonfree_edges"] + lp["num_free_edges"] + token_index, -float(max_pack)))
            violations.append((violation, full_key, entries, 0.0))
    return violations


def separate_global_token_packing_cut_specs(
    lp,
    f_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
):
    """Separate weighted corpus-level packing cuts for a token colour.

    For token colour tau:

        sum_w freq[w] * sum_{e in occurrences(w,tau)} f[e]
            <= sum_w freq[w] * max_pack(w,tau) * t[tau]

    This is the weighted sum of all same-colour word-packing inequalities for
    tau, but it is much more compact and targets objective-relevant violations.
    """

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    for token_index, edge_indices in lp["global_token_occurrences"].items():
        full_key = ("global_token_packing", token_index)
        if full_key in existing_cut_keys:
            continue
        coefficients = lp["global_token_occurrence_weights"][token_index]
        lhs_value = float(np.dot(coefficients, f_values[edge_indices]))
        max_pack = float(lp["global_token_max_pack"][token_index])
        rhs_value = max_pack * float(t_values[token_index])
        violation = lhs_value - rhs_value
        if violation <= tolerance:
            continue
        entries = [(edge_idx, coeff) for edge_idx, coeff in zip(edge_indices, coefficients)]
        entries.append((t_offset + token_index, -max_pack))
        violations.append((violation, full_key, entries, 0.0))

    return violations


def separate_global_rank_packing_cut_specs(
    lp,
    f_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    rank: int,
    max_words: int = 1000,
    top_colors_per_word: int = 8,
):
    """Separate weighted corpus-level small-rank interval-capacity cuts."""

    rank_data = {}
    for word_idx in rank_suspicious_words(lp, f_values, max_words=max_words):
        edge_indices = lp["word_nonfree_edges"].get(word_idx, [])
        by_color = defaultdict(list)
        scores = defaultdict(float)
        word_weight = float(lp["word_weights"][word_idx])
        for edge_idx in edge_indices:
            value = float(f_values[edge_idx])
            if value <= tolerance:
                continue
            info = lp["nonfree_edge_info"][edge_idx]
            token_idx = info["token_index"]
            length = max(1, info["end"] - info["start"])
            by_color[token_idx].append(edge_idx)
            scores[token_idx] += word_weight * length * value

        colors = [
            token_idx
            for token_idx, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_colors_per_word]
        ]
        for color_tuple in combinations(colors, rank):
            color_tuple = tuple(sorted(color_tuple))
            data = rank_data.setdefault(
                color_tuple,
                {
                    "entries": [],
                    "capacities": {mask: 0.0 for mask in range(1 << rank)},
                },
            )
            for token_idx in color_tuple:
                color_edges = by_color[token_idx]
                data["entries"].extend(
                    (
                        edge_idx,
                        word_weight
                        * max(1, lp["nonfree_edge_info"][edge_idx]["end"] - lp["nonfree_edge_info"][edge_idx]["start"]),
                    )
                    for edge_idx in color_edges
                )

            for mask in range(1, 1 << rank):
                mask_edges = []
                for bit, token_idx in enumerate(color_tuple):
                    if mask & (1 << bit):
                        mask_edges.extend(by_color[token_idx])
                data["capacities"][mask] += word_weight * color_window_capacity(
                    lp,
                    [
                        (
                            edge_idx,
                            max(1, lp["nonfree_edge_info"][edge_idx]["end"] - lp["nonfree_edge_info"][edge_idx]["start"]),
                        )
                        for edge_idx in mask_edges
                    ],
                )

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    family_name = "global_pair_packing" if rank == 2 else f"global_rank{rank}_packing"
    for color_tuple, data in rank_data.items():
        full_key = (family_name, *color_tuple)
        if full_key in existing_cut_keys:
            continue
        capacities = data["capacities"]
        bound = modular_upper_bound_for_capacities(capacities, t_values, color_tuple)
        if bound is None:
            continue
        alpha, beta_values, rhs_at_current = bound
        usage = sum(coeff * float(f_values[edge_idx]) for edge_idx, coeff in data["entries"])
        violation = usage - rhs_at_current
        if violation <= tolerance:
            continue
        entries = list(data["entries"])
        entries.extend(
            (t_offset + token_idx, -float(beta))
            for token_idx, beta in zip(color_tuple, beta_values)
            if abs(beta) > 1e-12
        )
        violations.append((violation, full_key, entries, float(alpha)))

    return violations


def separate_window_overlap_cut_specs(
    lp,
    f_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 250,
    max_rank: int = 4,
    window_lengths: tuple[int, ...] = (4, 6, 8, 10, 12),
):
    """Separate small-rank overlap capacity cuts on suspicious byte windows.

    For a word window W and selected token colours C, let y be fractional
    clipped window coverage from edges of colours C. For every subset U of C,
    compute cap(U), the maximum clipped coverage a real non-overlapping packing
    can obtain using only colours U. A modular upper bound on cap(U) gives:

        y <= alpha + sum_{tau in C} beta_tau * t_tau

    The modular bound is optimized for the current fractional t values.
    """

    suspicious_words = rank_suspicious_words(lp, f_values, max_words=max_words)
    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    for word_idx in suspicious_words:
        word_len = lp["word_lengths"][word_idx]
        if word_len <= 1:
            continue
        word_edge_indices = lp["word_nonfree_edges"].get(word_idx, [])
        if len(word_edge_indices) == 0:
            continue

        for window_len in window_lengths:
            if window_len > word_len:
                continue
            for start in range(0, word_len - window_len + 1):
                end = start + window_len
                color_scores = defaultdict(float)
                window_edges = []
                for edge_idx in word_edge_indices:
                    info = lp["nonfree_edge_info"][edge_idx]
                    overlap = interval_overlap(info["start"], info["end"], start, end)
                    if overlap <= 0:
                        continue
                    token_idx = info["token_index"]
                    value = float(f_values[edge_idx])
                    if value <= tolerance:
                        continue
                    color_scores[token_idx] += overlap * value
                    window_edges.append((edge_idx, token_idx, overlap))

                if len(color_scores) < 2:
                    continue

                selected_tokens = tuple(
                    token_idx
                    for token_idx, _ in sorted(
                        color_scores.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:max_rank]
                )
                full_key_base = ("window_overlap", word_idx, start, end, selected_tokens)
                if any(key[:5] == full_key_base for key in existing_cut_keys):
                    continue

                selected_set = set(selected_tokens)
                selected_edges = [
                    (edge_idx, token_idx, overlap)
                    for edge_idx, token_idx, overlap in window_edges
                    if token_idx in selected_set
                ]
                if not selected_edges:
                    continue

                fractional_usage = sum(
                    overlap * float(f_values[edge_idx])
                    for edge_idx, _, overlap in selected_edges
                )
                capacities = subset_window_capacities(lp, selected_edges, selected_tokens)
                bound = modular_upper_bound_for_capacities(capacities, t_values, selected_tokens)
                if bound is None:
                    continue
                alpha, beta_values, rhs_at_current = bound
                violation = fractional_usage - rhs_at_current
                if violation <= tolerance:
                    continue

                beta_key = tuple(round(float(beta), 8) for beta in beta_values)
                full_key = (*full_key_base, beta_key)
                if full_key in existing_cut_keys:
                    continue

                entries = [(edge_idx, float(overlap)) for edge_idx, _, overlap in selected_edges]
                entries.extend(
                    (t_offset + token_idx, -float(beta))
                    for token_idx, beta in zip(selected_tokens, beta_values)
                    if abs(beta) > 1e-12
                )
                violations.append((violation, full_key, entries, float(alpha)))

    return violations


def separate_window_pair_cut_specs(
    lp,
    f_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 500,
    top_colors: int = 8,
    window_lengths: tuple[int, ...] = (4, 6, 8, 10, 12),
):
    """Separate pairwise local overlap cuts.

    For two colours A,B in a window, compute cA, cB, cAB where cAB is the max
    compatible clipped coverage using occurrences of either colour. It then
    fits the best modular upper bound at the current fractional t, subject to
    validity for the four binary activation subsets.
    """

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    for word_idx in rank_suspicious_words(lp, f_values, max_words=max_words):
        word_len = lp["word_lengths"][word_idx]
        if word_len <= 1:
            continue
        word_edge_indices = lp["word_nonfree_edges"].get(word_idx, [])
        for window_len in window_lengths:
            if window_len > word_len:
                continue
            for start in range(0, word_len - window_len + 1):
                end = start + window_len
                by_color = defaultdict(list)
                scores = defaultdict(float)
                for edge_idx in word_edge_indices:
                    info = lp["nonfree_edge_info"][edge_idx]
                    overlap = interval_overlap(info["start"], info["end"], start, end)
                    if overlap <= 0 or f_values[edge_idx] <= tolerance:
                        continue
                    token_idx = info["token_index"]
                    by_color[token_idx].append((edge_idx, overlap))
                    scores[token_idx] += float(overlap * f_values[edge_idx])
                colors = [
                    token_idx
                    for token_idx, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_colors]
                ]
                for left_pos, left in enumerate(colors):
                    for right in colors[left_pos + 1 :]:
                        full_key = ("window_pair", word_idx, start, end, left, right)
                        if full_key in existing_cut_keys:
                            continue
                        selected_tokens = (left, right)
                        selected_edges = [
                            (edge_idx, token_idx, overlap)
                            for token_idx in selected_tokens
                            for edge_idx, overlap in by_color[token_idx]
                        ]
                        capacities = subset_window_capacities(lp, selected_edges, selected_tokens)
                        bound = modular_upper_bound_for_capacities(capacities, t_values, selected_tokens)
                        if bound is None:
                            continue
                        alpha, beta_values, rhs = bound
                        usage = sum(
                            overlap * float(f_values[edge_idx])
                            for edge_idx, _, overlap in selected_edges
                        )
                        violation = usage - rhs
                        if violation <= tolerance:
                            continue

                        beta_key = tuple(round(float(beta), 8) for beta in beta_values)
                        keyed_full_key = (*full_key, beta_key)
                        if keyed_full_key in existing_cut_keys:
                            continue
                        entries = [(edge_idx, float(overlap)) for edge_idx, _, overlap in selected_edges]
                        entries.extend(
                            (t_offset + token_idx, -float(beta))
                            for token_idx, beta in zip(selected_tokens, beta_values)
                            if abs(beta) > 1e-12
                        )
                        cut_rhs = float(alpha)
                        violations.append((violation, keyed_full_key, entries, cut_rhs))
    return violations


def color_window_capacity(lp, edge_overlaps) -> float:
    intervals = []
    for edge_idx, overlap in edge_overlaps:
        info = lp["nonfree_edge_info"][edge_idx]
        intervals.append((info["start"], info["end"], float(overlap)))
    return weighted_interval_capacity(intervals)


def rank_suspicious_words(lp, f_values, *, max_words: int):
    scores = []
    for word_idx, edge_indices in lp["word_nonfree_edges"].items():
        score = 0.0
        for edge_idx in edge_indices:
            value = float(f_values[edge_idx])
            if 1e-9 < value < 1.0 - 1e-9:
                info = lp["nonfree_edge_info"][edge_idx]
                score += min(value, 1.0 - value) * max(1, info["end"] - info["start"])
        if score > 0:
            scores.append((score * lp["word_weights"][word_idx], word_idx))
    scores.sort(reverse=True)
    return [word_idx for _, word_idx in scores[:max_words]]


def subset_window_capacities(lp, selected_edges, selected_tokens):
    token_position = {token_idx: bit for bit, token_idx in enumerate(selected_tokens)}
    capacities = {}
    for mask in range(1 << len(selected_tokens)):
        intervals = []
        for edge_idx, token_idx, overlap in selected_edges:
            bit = token_position[token_idx]
            if mask & (1 << bit):
                info = lp["nonfree_edge_info"][edge_idx]
                intervals.append((info["start"], info["end"], float(overlap)))
        capacities[mask] = weighted_interval_capacity(intervals)
    return capacities


def weighted_interval_capacity(intervals) -> float:
    if not intervals:
        return 0.0
    sorted_intervals = sorted(intervals, key=lambda item: (item[1], item[0]))
    ends = [end for _, end, _ in sorted_intervals]
    dp = [0.0] * (len(sorted_intervals) + 1)
    for idx, (start, _, weight) in enumerate(sorted_intervals, start=1):
        prev = np.searchsorted(ends, start, side="right")
        dp[idx] = max(dp[idx - 1], dp[prev] + weight)
    return dp[-1]


def modular_upper_bound_for_capacities(capacities, t_values, selected_tokens):
    rank = len(selected_tokens)
    objective = np.array([1.0, *[float(t_values[token_idx]) for token_idx in selected_tokens]])
    a_ub = []
    b_ub = []
    for mask, capacity in capacities.items():
        row = [-1.0]
        for bit in range(rank):
            row.append(-1.0 if mask & (1 << bit) else 0.0)
        a_ub.append(row)
        b_ub.append(-float(capacity))

    bounds = [(0.0, None)] * (rank + 1)
    result = linprog(
        c=objective,
        A_ub=np.array(a_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None
    alpha = float(result.x[0])
    beta_values = np.array(result.x[1:], dtype=float)
    rhs_at_current = float(objective @ result.x)
    return alpha, beta_values, rhs_at_current


def interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def separate_path_config_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_paths_per_word: int = 10000,
):
    """Separate conservative word-path configuration cuts.

    For a word and threshold K, find a hitting set S of token colours that
    intersects every complete path with <= K tokens. Then the ILP-valid cut is:

        word_cost + K * sum_{tau in S} t_tau >= K + 1

    If no S token is active integrally, all <=K paths are blocked; otherwise the
    RHS drops to at most 1, which every non-empty word path satisfies.
    """

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    for word_idx, nonfree_edges in lp["word_nonfree_edges"].items():
        free_edges = lp["word_free_edges"].get(word_idx, [])
        word_cost = float(f_values[nonfree_edges].sum() + g_values[free_edges].sum())
        if word_cost <= 1.0 + tolerance:
            continue

        k_value = int(np.floor(word_cost + 1e-9))
        if k_value <= 0:
            continue

        full_key_prefix = ("path_config", word_idx, k_value)
        if any(key[:3] == full_key_prefix for key in existing_cut_keys):
            continue

        path_token_sets = enumerate_short_path_token_sets(
            lp,
            word_idx,
            k_value,
            max_paths=max_paths_per_word,
        )
        if path_token_sets is None or not path_token_sets:
            continue
        if any(len(token_set) == 0 for token_set in path_token_sets):
            continue

        hitting_set = greedy_weighted_hitting_set(path_token_sets, t_values)
        if not hitting_set:
            continue

        t_sum = float(t_values[list(hitting_set)].sum())
        lhs_value = word_cost + k_value * t_sum
        rhs_value = float(k_value + 1)
        violation = rhs_value - lhs_value
        if violation <= tolerance:
            continue

        full_key = (*full_key_prefix, tuple(sorted(hitting_set)))
        if full_key in existing_cut_keys:
            continue

        entries = []
        entries.extend((edge_idx, -1.0) for edge_idx in nonfree_edges)
        entries.extend((num_f + edge_idx, -1.0) for edge_idx in free_edges)
        entries.extend((t_offset + token_idx, -float(k_value)) for token_idx in hitting_set)
        violations.append((violation, full_key, entries, -rhs_value))

    return violations


def separate_path_multicover_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_paths_per_word: int = 10000,
):
    """Separate path multicover cuts without auxiliary variables.

    If every complete path with <= K tokens uses at least r token colours from S,
    then the following is ILP-valid:

        word_cost + (K / r) * sum_{tau in S} t_tau >= K + 1

    When an integral solution has a <=K path, all colours on that path are
    active, so sum_S t_tau >= r. Otherwise the word_cost term is already >=K+1.
    """

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    for word_idx, nonfree_edges in lp["word_nonfree_edges"].items():
        free_edges = lp["word_free_edges"].get(word_idx, [])
        word_cost = float(f_values[nonfree_edges].sum() + g_values[free_edges].sum())
        if word_cost <= 1.0 + tolerance:
            continue

        k_value = int(np.floor(word_cost + 1e-9))
        if k_value <= 0:
            continue

        path_token_sets = enumerate_short_path_token_sets(
            lp,
            word_idx,
            k_value,
            max_paths=max_paths_per_word,
        )
        if path_token_sets is None or not path_token_sets:
            continue
        if any(len(token_set) == 0 for token_set in path_token_sets):
            continue

        selected_tokens, min_cover = greedy_low_density_multicover(path_token_sets, t_values)
        if not selected_tokens or min_cover <= 0:
            continue

        coefficient = float(k_value / min_cover)
        t_sum = float(t_values[list(selected_tokens)].sum())
        lhs_value = word_cost + coefficient * t_sum
        rhs_value = float(k_value + 1)
        violation = rhs_value - lhs_value
        if violation <= tolerance:
            continue

        full_key = ("path_multicover", word_idx, k_value, min_cover, tuple(sorted(selected_tokens)))
        if full_key in existing_cut_keys:
            continue

        entries = []
        entries.extend((edge_idx, -1.0) for edge_idx in nonfree_edges)
        entries.extend((num_f + edge_idx, -1.0) for edge_idx in free_edges)
        entries.extend((t_offset + token_idx, -coefficient) for token_idx in selected_tokens)
        violations.append((violation, full_key, entries, -rhs_value))

    return violations


def separate_word_path_cover_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 300,
    max_paths_per_word: int = 20000,
    max_hitting_tokens: int = 12,
):
    """Separate word-level path cover cuts.

    If S hits every path with <=K tokens through a word, then in any integral
    solution:

        word_cost >= K + 1 - K * sum_{tau in S} t_tau

    Unlike path_config, this tries several cheap partial hitting sets found by
    repeatedly selecting low-t-value colours, not just a complete greedy set.
    """

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    for word_idx in rank_suspicious_words(lp, f_values, max_words=max_words):
        nonfree_edges = lp["word_nonfree_edges"].get(word_idx, [])
        free_edges = lp["word_free_edges"].get(word_idx, [])
        word_cost = float(f_values[nonfree_edges].sum() + g_values[free_edges].sum())
        if word_cost <= 1.0 + tolerance:
            continue

        k_value = int(np.floor(word_cost + 1e-9))
        if k_value <= 0:
            continue

        path_token_sets = enumerate_short_path_token_sets(
            lp,
            word_idx,
            k_value,
            max_paths=max_paths_per_word,
        )
        if path_token_sets is None or not path_token_sets:
            continue
        if any(len(token_set) == 0 for token_set in path_token_sets):
            continue

        for hitting_set in candidate_low_weight_hitting_sets(
            path_token_sets,
            t_values,
            max_hitting_tokens=max_hitting_tokens,
        ):
            full_key = ("word_path_cover", word_idx, k_value, tuple(sorted(hitting_set)))
            if full_key in existing_cut_keys:
                continue
            t_sum = float(t_values[list(hitting_set)].sum())
            lhs_value = word_cost + k_value * t_sum
            rhs_value = float(k_value + 1)
            violation = rhs_value - lhs_value
            if violation <= tolerance:
                continue

            entries = []
            entries.extend((edge_idx, -1.0) for edge_idx in nonfree_edges)
            entries.extend((num_f + edge_idx, -1.0) for edge_idx in free_edges)
            entries.extend((t_offset + token_idx, -float(k_value)) for token_idx in hitting_set)
            violations.append((violation, full_key, entries, -rhs_value))
            break

    return violations


def enumerate_short_path_token_sets(lp, word_idx: int, max_length: int, *, max_paths: int):
    by_start = defaultdict(list)
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        info = lp["nonfree_edge_info"][edge_idx]
        by_start[info["start"]].append((info["end"], info["token_index"]))
    for edge_idx in lp["word_free_edges"].get(word_idx, []):
        info = lp["free_edge_info"][edge_idx]
        by_start[info["start"]].append((info["end"], None))

    target = lp["word_lengths"][word_idx]
    results = []

    def visit(position: int, remaining: int, token_set: frozenset[int]):
        if len(results) > max_paths:
            return
        if position == target:
            results.append(token_set)
            return
        if remaining == 0:
            return
        for end, token_idx in by_start.get(position, []):
            next_set = token_set if token_idx is None else token_set | frozenset((token_idx,))
            visit(end, remaining - 1, next_set)

    visit(0, max_length, frozenset())
    if len(results) > max_paths:
        return None
    return results


def greedy_weighted_hitting_set(path_token_sets, t_values):
    uncovered = [set(token_set) for token_set in path_token_sets]
    hitting_set = set()
    while uncovered:
        scores = defaultdict(int)
        for token_set in uncovered:
            for token_idx in token_set:
                scores[token_idx] += 1
        if not scores:
            return set()
        best_token = max(
            scores,
            key=lambda token_idx: (scores[token_idx] / (float(t_values[token_idx]) + 1e-9), scores[token_idx]),
        )
        hitting_set.add(best_token)
        uncovered = [token_set for token_set in uncovered if best_token not in token_set]
    return hitting_set


def candidate_low_weight_hitting_sets(path_token_sets, t_values, *, max_hitting_tokens: int):
    uncovered = [set(token_set) for token_set in path_token_sets]
    selected = set()
    candidates = []

    while uncovered and len(selected) < max_hitting_tokens:
        scores = defaultdict(int)
        for token_set in uncovered:
            for token_idx in token_set:
                scores[token_idx] += 1
        if not scores:
            break
        best_token = max(
            scores,
            key=lambda token_idx: (scores[token_idx] / (float(t_values[token_idx]) + 1e-9), scores[token_idx]),
        )
        selected.add(best_token)
        uncovered = [token_set for token_set in uncovered if best_token not in token_set]
        if not uncovered:
            candidates.append(set(selected))
            break

    if candidates:
        return candidates
    return []


def greedy_low_density_multicover(path_token_sets, t_values):
    token_universe = sorted(set().union(*path_token_sets))
    if not token_universe:
        return set(), 0

    selected = set()
    cover_counts = [0] * len(path_token_sets)
    best_selected = set()
    best_min_cover = 0
    best_density = float("inf")

    while True:
        best_token = None
        best_score = None
        for token_idx in token_universe:
            if token_idx in selected:
                continue
            new_counts = [
                count + (1 if token_idx in token_set else 0)
                for count, token_set in zip(cover_counts, path_token_sets)
            ]
            new_min_cover = min(new_counts)
            total_weight = float(t_values[list(selected)].sum()) + float(t_values[token_idx])
            density = total_weight / new_min_cover if new_min_cover > 0 else float("inf")
            total_coverage = sum(new_counts)
            score = (-density, new_min_cover, total_coverage, -float(t_values[token_idx]))
            if best_score is None or score > best_score:
                best_score = score
                best_token = token_idx

        if best_token is None:
            break

        selected.add(best_token)
        cover_counts = [
            count + (1 if best_token in token_set else 0)
            for count, token_set in zip(cover_counts, path_token_sets)
        ]
        min_cover = min(cover_counts)
        if min_cover > 0:
            density = float(t_values[list(selected)].sum()) / min_cover
            if density + 1e-12 < best_density:
                best_density = density
                best_selected = set(selected)
                best_min_cover = min_cover

        if len(selected) == len(token_universe):
            break

    return best_selected, best_min_cover


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

    tokens = sorted(set(token for row in token_lists for token in row), key=lambda token: token.token)
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
    word_token_occurrences = defaultdict(list)
    word_token_intervals = defaultdict(list)
    word_nonfree_edges = defaultdict(list)
    word_free_edges = defaultdict(list)
    global_token_occurrences = defaultdict(list)
    global_token_occurrence_weights = defaultdict(list)
    nonfree_edge_info = []
    free_edge_info = []
    word_lengths = []
    word_weights = []
    vertex_offset = edge_offset = free_edge_offset = 0

    for word_idx, (edges, free_edges, weight, num_vertices) in enumerate(
        zip(edges_list, free_edges_list, edge_weights, num_vertices_list)
    ):
        word_lengths.append(num_vertices - 1)
        word_weights.append(float(weight))
        for local_idx, edge in enumerate(edges):
            edge_idx = edge_offset + local_idx
            edge_token_index = token_index[edge.token]
            word_nonfree_edges[word_idx].append(edge_idx)
            a_rows.extend([vertex_offset + edge.start, vertex_offset + edge.end])
            a_cols.extend([edge_idx, edge_idx])
            a_data.extend([1.0, -1.0])
            m_rows.append(edge_idx)
            m_cols.append(edge_token_index)
            m_data.append(1.0)
            word_token_key = (word_idx, edge_token_index)
            word_token_occurrences[word_token_key].append(edge_idx)
            word_token_intervals[word_token_key].append((edge.start, edge.end))
            global_token_occurrences[edge_token_index].append(edge_idx)
            global_token_occurrence_weights[edge_token_index].append(float(weight))
            nonfree_edge_info.append(
                {
                    "word_idx": word_idx,
                    "start": edge.start,
                    "end": edge.end,
                    "token": edge.token,
                    "token_index": edge_token_index,
                }
            )
            for boundary in range(edge.start, edge.end):
                boundary_crossings[(word_idx, boundary, edge_token_index)].append(edge_idx)

        for local_idx, edge in enumerate(free_edges):
            free_edge_idx = free_edge_offset + local_idx
            word_free_edges[word_idx].append(free_edge_idx)
            b_rows.extend([vertex_offset + edge.start, vertex_offset + edge.end])
            b_cols.extend([free_edge_idx, free_edge_idx])
            b_data.extend([1.0, -1.0])
            free_edge_info.append(
                {
                    "word_idx": word_idx,
                    "start": edge.start,
                    "end": edge.end,
                    "token": edge.token,
                }
            )

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

    word_token_max_pack = {
        key: max_non_overlapping_intervals(intervals)
        for key, intervals in word_token_intervals.items()
    }
    global_token_max_pack = defaultdict(float)
    for (word_idx, token_index), max_pack in word_token_max_pack.items():
        global_token_max_pack[token_index] += float(edge_weights[word_idx]) * float(max_pack)

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
        "word_token_occurrences": dict(word_token_occurrences),
        "word_token_max_pack": word_token_max_pack,
        "global_token_occurrences": {
            key: np.array(value, dtype=int) for key, value in global_token_occurrences.items()
        },
        "global_token_occurrence_weights": {
            key: np.array(value, dtype=float)
            for key, value in global_token_occurrence_weights.items()
        },
        "global_token_max_pack": dict(global_token_max_pack),
        "word_nonfree_edges": {key: np.array(value, dtype=int) for key, value in word_nonfree_edges.items()},
        "word_free_edges": {key: np.array(value, dtype=int) for key, value in word_free_edges.items()},
        "nonfree_edge_info": nonfree_edge_info,
        "free_edge_info": free_edge_info,
        "word_lengths": word_lengths,
        "word_weights": word_weights,
    }


def max_non_overlapping_intervals(intervals) -> int:
    count = 0
    end_so_far = -1
    for start, end in sorted(intervals, key=lambda interval: (interval[1], interval[0])):
        if start >= end_so_far:
            count += 1
            end_so_far = end
    return count


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
