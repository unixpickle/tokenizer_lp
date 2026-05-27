from __future__ import annotations

import multiprocessing as mp
import os
import json
import logging
import time
import hashlib
import heapq
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields, replace
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
class CutSeparationConfig:
    word_support_max_words: int = 2000
    word_support_max_rank: int = 20
    word_support_max_paths: int = 100000
    short_word_hull_max_words: int = 2000
    short_word_hull_max_length: int = 10
    short_word_hull_max_rank: int = 16
    short_word_full_hull_max_words: int = 250
    short_word_full_hull_max_length: int = 8
    short_word_full_hull_max_colors: int = 64
    short_word_pair_hull_max_words: int = 500
    short_word_pair_hull_max_length: int = 12
    short_word_pair_hull_max_colors: int = 96
    short_word_pair_hull_max_pair_rows: int = 250000
    short_word_pair_hull_max_pairs: int = 800
    short_word_pair_hull_top_words_per_color: int = 36
    short_word_pair_hull_candidate_word_multiplier: float = 1.0
    short_word_pair_hull_candidate_top_words_multiplier: float = 1.0
    short_word_pair_hull_candidate_strategy: str = "score"
    short_word_pair_hull_candidate_random_seed: int = 0
    short_word_pair_hull_pruning: str = "full"
    short_word_pair_hull_workers: int = 0
    short_word_pair_hull_batch_size: int = 32
    short_word_pair_hull_min_fractional_shared_colors: int = 1
    short_word_pair_hull_cache_max_entries: int = 500000
    short_word_pair_hull_cache_value_quantum: float = 1e-4
    short_word_pair_hull_solution_cache: dict[str, dict | None] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    short_word_pair_template_max_words: int = 700
    short_word_pair_template_max_length: int = 12
    short_word_pair_template_candidate_word_multiplier: float = 1.0
    short_word_pair_template_top_supports_per_shape: int = 24
    short_word_pair_template_max_chain_edges: int = 6
    short_word_pair_template_max_cuts: int = 100000
    short_word_triple_hull_max_words: int = 700
    short_word_triple_hull_max_length: int = 12
    short_word_triple_hull_max_rows: int = 200000
    short_word_triple_hull_max_triples: int = 30000
    short_word_triple_hull_top_words_per_color: int = 48
    short_word_triple_hull_candidate_word_multiplier: float = 1.0
    short_word_triple_hull_candidate_top_words_multiplier: float = 1.0
    short_word_triple_hull_candidate_sample: int = 250000
    short_word_triple_hull_candidate_random_seed: int = 0
    short_word_triple_hull_token_mode: str = "at_least_two"
    short_word_triple_hull_min_fractional_colors: int = 2
    short_word_triple_hull_workers: int = 0
    short_word_triple_hull_batch_size: int = 32
    short_word_triple_template_max_words: int = 700
    short_word_triple_template_max_length: int = 12
    short_word_triple_template_candidate_word_multiplier: float = 1.0
    short_word_triple_template_top_supports_per_shape: int = 24
    short_word_triple_template_max_cuts: int = 100000
    short_word_triple_template_validate: bool = False
    run_all_cut_families: bool = False


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
    fractional_colors: int
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
    cut_config: CutSeparationConfig | None = None,
    lp_solution_cache_dir: str | Path | None = None,
    lp_solver: str = "highspy",
    resume: bool = True,
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
        cut_config=cut_config,
        lp_solution_cache_dir=lp_solution_cache_dir,
        lp_solver=lp_solver,
        resume_state_dir=(Path(output_dir) / "training_state") if output_dir is not None and resume else None,
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
    cut_config: CutSeparationConfig | None = None,
    lp_solution_cache_dir: str | Path | None = None,
    lp_solver: str = "highspy",
    resume_state_dir: str | Path | None = None,
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
    cut_config = cut_config or CutSeparationConfig()

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
    resume_checkpoint = None
    active_cut_matrix = sp.csr_matrix((0, lp["c"].shape[0]), dtype=float)
    active_cut_rhs = np.array([], dtype=float)
    next_iteration = 0
    state_dir = Path(resume_state_dir) if resume_state_dir is not None else None
    highs_basis_path = state_dir / "highs_basis.bas" if state_dir is not None else None
    state_key = lp_training_state_key(
        word_counts=word_counts,
        num_allowed_tokens=num_allowed_tokens,
        min_token_count=min_token_count,
        max_token_length=max_token_length,
        cut_rounds=cut_rounds,
        cuts_per_round=cuts_per_round,
        cut_tolerance=cut_tolerance,
        cut_families=cut_families,
        cut_config=cut_config,
        lp=lp,
    )
    if state_dir is not None:
        resume_checkpoint = load_lp_training_checkpoint(state_dir, state_key, num_vars=lp["c"].shape[0])
        if resume_checkpoint is not None:
            if resume_checkpoint["completed"]:
                LOGGER.info("Loaded completed LP training checkpoint from %s", state_dir)
                return resume_checkpoint["final_candidates"]
            existing_cut_keys = set(resume_checkpoint["existing_cut_keys"])
            iterations = list(resume_checkpoint["iterations"])
            final_candidates = resume_checkpoint["final_candidates"]
            cut_config.short_word_pair_hull_solution_cache.update(
                resume_checkpoint["short_word_pair_hull_solution_cache"]
            )
            active_cut_matrix = resume_checkpoint["cut_matrix"]
            active_cut_rhs = resume_checkpoint["cut_rhs"]
            next_iteration = int(resume_checkpoint["next_iteration"])
            if active_cut_matrix.shape[0]:
                a_ub = sp.vstack([a_ub, active_cut_matrix], format="csr")
                b_ub = np.concatenate([b_ub, active_cut_rhs])
            LOGGER.info(
                "Resuming LP training from %s at iteration %d with %d active cuts",
                state_dir,
                next_iteration,
                len(existing_cut_keys),
            )
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
            basis_path=highs_basis_path,
        )
    elif lp_solver != "scipy":
        raise ValueError(f"Unsupported LP solver {lp_solver!r}. Expected 'highspy' or 'scipy'.")

    for iteration in range(next_iteration, cut_rounds + 1):
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
        num_f = lp["num_nonfree_edges"]
        num_g = lp["num_free_edges"]
        t_values = solution.x[num_f + num_g :]
        fractional_colors = int(np.count_nonzero((t_values > cut_tolerance) & (t_values < 1.0 - cut_tolerance)))
        candidates.objective_value = float(solution.fun)
        candidates.token_count_lower_bound = float(solution.fun)
        candidates.solve_seconds = solve_seconds
        candidates.status = str(solution.message)

        iteration_result = LpSolveIteration(
            iteration=iteration,
            objective_value=float(solution.fun),
            token_count_lower_bound=float(solution.fun),
            solve_seconds=solve_seconds,
            status=str(solution.message),
            total_cuts=len(existing_cut_keys),
            added_cuts=-1,
            max_violation=float("nan"),
            fractional_colors=fractional_colors,
            candidates=list(candidates),
        )
        iterations.append(iteration_result)
        LOGGER.info(
            "LP iteration %d solved in %.3fs: objective=%.3f nonzero_tokens=%d "
            "fractional_colors=%d active_cuts=%d",
            iteration,
            solve_seconds,
            solution.fun,
            len(candidates),
            fractional_colors,
            len(existing_cut_keys),
        )
        if iteration_callback is not None:
            iteration_callback(iteration_result)

        final_candidates = candidates
        added_cuts = 0
        max_violation = 0.0
        cut_matrix = None
        cut_keys: list[tuple[int, int, int]] = []
        if iteration < cut_rounds:
            LOGGER.info("LP iteration %d starting cut separation", iteration)
            cut_matrix, cut_rhs, cut_keys, max_violation = separate_cuts(
                lp,
                solution.x,
                existing_cut_keys=existing_cut_keys,
                max_cuts=cuts_per_round,
                tolerance=cut_tolerance,
                families=cut_families,
                config=cut_config,
                separation_round=iteration,
            )
            added_cuts = len(cut_keys)
            iteration_result = replace(
                iteration_result,
                added_cuts=added_cuts,
                max_violation=max_violation,
            )
            iterations[-1] = iteration_result
            LOGGER.info(
                "LP iteration %d separation complete: active_cuts=%d next_cuts=%d max_cut_violation=%.6g",
                iteration,
                len(existing_cut_keys),
                added_cuts,
                max_violation,
            )

        if added_cuts == 0:
            final_candidates.iterations = iterations
            if state_dir is not None:
                if highs_solver is not None:
                    highs_solver.save_basis(highs_basis_path)
                save_lp_training_checkpoint(
                    state_dir,
                    state_key=state_key,
                    next_iteration=iteration + 1,
                    existing_cut_keys=existing_cut_keys,
                    cut_matrix=active_cut_matrix,
                    cut_rhs=active_cut_rhs,
                    iterations=iterations,
                    final_candidates=final_candidates,
                    latest_solution=solution.x,
                    short_word_pair_hull_solution_cache=cut_config.short_word_pair_hull_solution_cache,
                    completed=True,
                )
            break

        existing_cut_keys.update(cut_keys)
        active_cut_matrix = sp.vstack([active_cut_matrix, cut_matrix], format="csr")
        active_cut_rhs = np.concatenate([active_cut_rhs, cut_rhs])
        if highs_solver is not None:
            highs_solver.add_ub_rows(cut_matrix, cut_rhs)
            if state_dir is not None:
                highs_solver.save_basis(highs_basis_path)
        else:
            a_ub = sp.vstack([a_ub, cut_matrix], format="csr")
            b_ub = np.concatenate([b_ub, cut_rhs])
        if state_dir is not None:
            save_lp_training_checkpoint(
                state_dir,
                state_key=state_key,
                next_iteration=iteration + 1,
                existing_cut_keys=existing_cut_keys,
                cut_matrix=active_cut_matrix,
                cut_rhs=active_cut_rhs,
                iterations=iterations,
                final_candidates=final_candidates,
                latest_solution=solution.x,
                short_word_pair_hull_solution_cache=cut_config.short_word_pair_hull_solution_cache,
                completed=False,
            )

    final_candidates.iterations = iterations
    return final_candidates


def lp_training_state_key(
    *,
    word_counts: Counter[str],
    num_allowed_tokens: int,
    min_token_count: int,
    max_token_length: int | None,
    cut_rounds: int,
    cuts_per_round: int,
    cut_tolerance: float,
    cut_families: tuple[str, ...],
    cut_config: CutSeparationConfig,
    lp,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"tokenizer-lp-training-state-v1")
    hasher.update(str(int(num_allowed_tokens)).encode("ascii"))
    hasher.update(str(int(min_token_count)).encode("ascii"))
    hasher.update(str(max_token_length).encode("ascii"))
    cut_payload = {
        "cut_rounds": int(cut_rounds),
        "cuts_per_round": int(cuts_per_round),
        "cut_tolerance": float(cut_tolerance),
        "cut_families": list(cut_families),
        "cut_config": cut_config_state_payload(cut_config),
    }
    hasher.update(json.dumps(cut_payload, sort_keys=True).encode("utf-8"))
    hasher.update(str(int(lp["num_tokens"])).encode("ascii"))
    hasher.update(str(int(lp["num_nonfree_edges"])).encode("ascii"))
    hasher.update(str(int(lp["num_free_edges"])).encode("ascii"))
    for word, count in sorted(word_counts.items()):
        encoded = word.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)
        hasher.update(int(count).to_bytes(8, "little", signed=False))
    return hasher.hexdigest()


def cut_config_state_payload(config: CutSeparationConfig) -> dict:
    payload = {}
    for config_field in fields(config):
        if config_field.name == "short_word_pair_hull_solution_cache":
            continue
        payload[config_field.name] = getattr(config, config_field.name)
    return payload


def load_lp_training_checkpoint(state_dir: Path, state_key: str, *, num_vars: int):
    state_path = state_dir / "state.json"
    cut_matrix_path = state_dir / "active_cuts.npz"
    cut_rhs_path = state_dir / "active_cut_rhs.npy"
    latest_solution_path = state_dir / "latest_solution.npy"
    if not state_path.exists():
        return None

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Ignoring unreadable LP training checkpoint %s: %s", state_path, exc)
        return None

    if payload.get("version") != 1:
        LOGGER.warning("Ignoring unsupported LP training checkpoint version in %s", state_path)
        return None
    state_key_matches = payload.get("state_key") == state_key

    try:
        if cut_matrix_path.exists():
            cut_matrix = sp.load_npz(cut_matrix_path).tocsr()
        else:
            cut_matrix = sp.csr_matrix((0, num_vars), dtype=float)
        if cut_rhs_path.exists():
            cut_rhs = np.load(cut_rhs_path, allow_pickle=False)
        else:
            cut_rhs = np.array([], dtype=float)
        if latest_solution_path.exists():
            latest_solution = np.load(latest_solution_path, allow_pickle=False)
        else:
            latest_solution = None
    except (OSError, ValueError) as exc:
        LOGGER.warning("Ignoring unreadable LP training cut checkpoint in %s: %s", state_dir, exc)
        return None

    if cut_matrix.shape[1] != num_vars or cut_matrix.shape[0] != cut_rhs.shape[0]:
        LOGGER.warning("Ignoring LP training checkpoint with incompatible cut dimensions: %s", state_dir)
        return None
    if latest_solution is not None and latest_solution.shape[0] != num_vars:
        LOGGER.warning("Ignoring LP training checkpoint with incompatible solution dimensions: %s", state_dir)
        return None
    if not state_key_matches:
        LOGGER.warning(
            "LP training checkpoint corpus/config hash differs, but dimensions match; "
            "resuming with existing cuts and current separation settings: %s",
            state_path,
        )

    final_candidates = candidate_list_from_json(payload.get("final_candidates") or [])
    final_metadata = payload.get("final_metadata") or {}
    final_candidates.objective_value = float(final_metadata.get("objective_value", float("nan")))
    final_candidates.token_count_lower_bound = float(final_metadata.get("token_count_lower_bound", float("nan")))
    final_candidates.solve_seconds = float(final_metadata.get("solve_seconds", float("nan")))
    final_candidates.status = str(final_metadata.get("status", "unknown"))
    final_candidates.iterations = [
        lp_solve_iteration_from_json(item)
        for item in payload.get("iterations", [])
    ]

    return {
        "completed": bool(payload.get("completed", False)),
        "next_iteration": int(payload.get("next_iteration", 0)),
        "existing_cut_keys": [
            tupleify_json_key(key)
            for key in payload.get("existing_cut_keys", [])
        ],
        "cut_matrix": cut_matrix,
        "cut_rhs": np.asarray(cut_rhs, dtype=float),
        "iterations": final_candidates.iterations,
        "final_candidates": final_candidates,
        "latest_solution": latest_solution,
        "short_word_pair_hull_solution_cache": pair_hull_solution_cache_from_json(
            payload.get("short_word_pair_hull_solution_cache") or {}
        ),
    }


def save_lp_training_checkpoint(
    state_dir: Path,
    *,
    state_key: str,
    next_iteration: int,
    existing_cut_keys: set[tuple],
    cut_matrix,
    cut_rhs,
    iterations: list[LpSolveIteration],
    final_candidates: CandidateList,
    latest_solution,
    short_word_pair_hull_solution_cache: dict[str, dict | None],
    completed: bool,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    cut_matrix = cut_matrix.tocsr()
    cut_rhs = np.asarray(cut_rhs, dtype=float)
    payload = {
        "version": 1,
        "state_key": state_key,
        "completed": bool(completed),
        "next_iteration": int(next_iteration),
        "existing_cut_keys": [jsonify_key(key) for key in sorted(existing_cut_keys, key=repr)],
        "iterations": [lp_solve_iteration_to_json(iteration) for iteration in iterations],
        "final_candidates": [possible_token_to_json(token) for token in final_candidates],
        "final_metadata": {
            "objective_value": float(getattr(final_candidates, "objective_value", float("nan"))),
            "token_count_lower_bound": float(
                getattr(final_candidates, "token_count_lower_bound", float("nan"))
            ),
            "solve_seconds": float(getattr(final_candidates, "solve_seconds", float("nan"))),
            "status": str(getattr(final_candidates, "status", "unknown")),
        },
        "short_word_pair_hull_solution_cache": pair_hull_solution_cache_to_json(
            short_word_pair_hull_solution_cache
        ),
    }

    atomic_save_sparse_npz(state_dir / "active_cuts.npz", cut_matrix)
    atomic_save_npy(state_dir / "active_cut_rhs.npy", cut_rhs)
    atomic_save_npy(state_dir / "latest_solution.npy", np.asarray(latest_solution, dtype=float))
    atomic_write_text(state_dir / "state.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    LOGGER.info(
        "Saved LP training checkpoint to %s: next_iteration=%d active_cuts=%d completed=%s",
        state_dir,
        next_iteration,
        cut_matrix.shape[0],
        completed,
    )


def lp_solve_iteration_to_json(iteration: LpSolveIteration) -> dict:
    return {
        "iteration": int(iteration.iteration),
        "objective_value": float(iteration.objective_value),
        "token_count_lower_bound": float(iteration.token_count_lower_bound),
        "solve_seconds": float(iteration.solve_seconds),
        "status": iteration.status,
        "total_cuts": int(iteration.total_cuts),
        "added_cuts": int(iteration.added_cuts),
        "max_violation": float(iteration.max_violation),
        "fractional_colors": int(iteration.fractional_colors),
        "candidates": [possible_token_to_json(token) for token in iteration.candidates],
    }


def lp_solve_iteration_from_json(payload: dict) -> LpSolveIteration:
    return LpSolveIteration(
        iteration=int(payload["iteration"]),
        objective_value=float(payload["objective_value"]),
        token_count_lower_bound=float(payload["token_count_lower_bound"]),
        solve_seconds=float(payload["solve_seconds"]),
        status=str(payload["status"]),
        total_cuts=int(payload["total_cuts"]),
        added_cuts=int(payload["added_cuts"]),
        max_violation=float(payload["max_violation"]),
        fractional_colors=int(payload["fractional_colors"]),
        candidates=list(candidate_list_from_json(payload.get("candidates", []))),
    )


def possible_token_to_json(token: possibleToken) -> dict:
    return {
        "token": token.token,
        "lp_value": float(token.lp_value),
        "instance_count": int(token.token_instance_count),
        "index": int(token.token_index),
    }


def candidate_list_from_json(payload: list[dict]) -> CandidateList:
    return CandidateList(
        possibleToken(
            token=str(item["token"]),
            lp_value=float(item["lp_value"]),
            instance_count=int(item["instance_count"]),
            index=int(item["index"]),
        )
        for item in payload
    )


def pair_hull_solution_cache_to_json(cache: dict[str, dict | None]) -> dict:
    encoded = {}
    for cache_key, value in cache.items():
        if value is None:
            encoded[cache_key] = None
            continue
        cut = value.get("cut")
        encoded[cache_key] = {
            "cut": cut_to_json(cut) if cut is not None else None,
            "build_seconds": float(value.get("build_seconds", 0.0)),
            "solve_seconds": float(value.get("solve_seconds", 0.0)),
            "cache_key": value.get("cache_key"),
        }
    return encoded


def pair_hull_solution_cache_from_json(payload: dict) -> dict[str, dict | None]:
    decoded = {}
    for cache_key, value in payload.items():
        if value is None:
            decoded[str(cache_key)] = None
            continue
        decoded[str(cache_key)] = {
            "cut": cut_from_json(value["cut"]) if value.get("cut") is not None else None,
            "build_seconds": float(value.get("build_seconds", 0.0)),
            "solve_seconds": float(value.get("solve_seconds", 0.0)),
            "cache_key": value.get("cache_key"),
        }
    return decoded


def cut_to_json(cut) -> list:
    violation, key, entries, rhs = cut
    return [
        float(violation),
        jsonify_key(key),
        [[int(col_idx), float(coefficient)] for col_idx, coefficient in entries],
        float(rhs),
    ]


def cut_from_json(payload: list):
    return (
        float(payload[0]),
        tupleify_json_key(payload[1]),
        [(int(col_idx), float(coefficient)) for col_idx, coefficient in payload[2]],
        float(payload[3]),
    )


def jsonify_key(value):
    if isinstance(value, tuple):
        return [jsonify_key(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def tupleify_json_key(value):
    if isinstance(value, list):
        return tuple(tupleify_json_key(item) for item in value)
    return value


def atomic_save_sparse_npz(path: Path, matrix) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        sp.save_npz(handle, matrix, compressed=True)
    os.replace(tmp_path, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(tmp_path, path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


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
        basis_path: str | Path | None = None,
        highs_threads: int = 0,
        highs_parallel: str = "on",
    ):
        import highspy

        self.highspy = highspy
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.basis_path = Path(basis_path) if basis_path is not None else None
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
        try:
            self.highs.resetGlobalScheduler(True)
        except RuntimeError:
            self.highs.resetGlobalScheduler(False)
        self.highs.setOptionValue("threads", int(highs_threads))
        self.highs.setOptionValue("parallel", highs_parallel)
        self.highs.setOptionValue("presolve", "on")
        self.highs.passModel(self._build_model())
        self.has_basis = False
        self._load_basis()

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
        self.has_basis = True
        self._save_cache(result)
        return result

    def save_basis(self, basis_path: str | Path | None = None) -> None:
        path = Path(basis_path) if basis_path is not None else self.basis_path
        if path is None or not self.has_basis:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
        status = self.highs.writeBasis(str(tmp_path))
        if status not in (self.highspy.HighsStatus.kOk, self.highspy.HighsStatus.kWarning) or not tmp_path.exists():
            LOGGER.warning("HiGHS failed to save basis to %s: %s", path, status)
            try:
                tmp_path.unlink()
            except OSError:
                pass
            return
        os.replace(tmp_path, path)
        LOGGER.info("Saved HiGHS basis to %s", path)

    def add_ub_rows(self, A_rows, b_rows) -> int | None:
        if A_rows is None:
            return None
        A_rows = A_rows.tocsr()
        b_rows = np.asarray(b_rows, dtype=float)
        if A_rows.shape[1] != self.c.shape[0]:
            raise ValueError(
                f"Cut matrix has {A_rows.shape[1]} columns, expected {self.c.shape[0]}."
            )
        if A_rows.shape[0] != b_rows.shape[0]:
            raise ValueError("Cut matrix row count must match cut RHS length.")
        if A_rows.shape[0] == 0:
            return None

        if self.has_basis:
            self.highs.setOptionValue("presolve", "off")
        start_row = self.A_ub.shape[0]
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
        return start_row

    def change_ub_row_bounds(self, row_idx: int, lower: float, upper: float) -> None:
        if self.has_basis:
            self.highs.setOptionValue("presolve", "off")
        highs_row_idx = self.A_eq.shape[0] + int(row_idx)
        self.highs.changeRowBounds(highs_row_idx, float(lower), float(upper))
        self.b_ub[row_idx] = float(upper)

    def change_col_bounds(self, col_idx: int, lower: float, upper: float) -> None:
        if self.has_basis:
            self.highs.setOptionValue("presolve", "off")
        self.highs.changeColBounds(int(col_idx), float(lower), float(upper))
        self.lb[col_idx] = float(lower)
        self.ub[col_idx] = float(upper)
        self.bounds[col_idx] = (float(lower), float(upper))

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

    def _load_basis(self) -> None:
        if self.basis_path is None or not self.basis_path.exists():
            return
        status = self.highs.readBasis(str(self.basis_path))
        if status != self.highspy.HighsStatus.kOk:
            LOGGER.warning("HiGHS failed to load basis from %s: %s", self.basis_path, status)
            return
        self.has_basis = True
        self.highs.setOptionValue("presolve", "off")
        LOGGER.info("Loaded HiGHS basis from %s", self.basis_path)

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
    config: CutSeparationConfig | None = None,
    separation_round: int = 0,
):
    if max_cuts <= 0:
        return None, np.array([], dtype=float), [], 0.0
    config = config or CutSeparationConfig()

    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    f_values = x_values[:num_f]
    t_values = x_values[num_f + num_g :]

    violations = PerFamilyCutList(max_per_extend=max_cuts)
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
    if "global_rank_count" in family_set:
        violations.extend(
            separate_global_rank_count_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "word_rank_count" in family_set:
        violations.extend(
            separate_word_rank_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                family_name="word_rank_count",
                edge_weight="count",
            )
        )
    if "word_rank_length" in family_set:
        violations.extend(
            separate_word_rank_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                family_name="word_rank_length",
                edge_weight="length",
            )
        )
    if "path_min_cover" in family_set:
        violations.extend(
            separate_path_min_cover_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "group_value" in family_set:
        violations.extend(
            separate_group_value_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "threshold_value" in family_set:
        violations.extend(
            separate_threshold_value_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "group_budget_value" in family_set:
        violations.extend(
            separate_group_budget_value_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "word_hull" in family_set:
        violations.extend(
            separate_word_hull_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                max_words=config.word_support_max_words,
                max_rank=min(config.word_support_max_rank, 16),
                max_paths=config.word_support_max_paths,
            )
        )
    if "short_word_hull" in family_set:
        violations.extend(
            separate_short_word_hull_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                max_words=config.short_word_hull_max_words,
                max_word_length=config.short_word_hull_max_length,
                max_rank=config.short_word_hull_max_rank,
                max_paths=config.word_support_max_paths,
            )
        )
    short_word_full_hull_violations = []
    if "short_word_full_hull" in family_set:
        short_word_full_hull_violations = separate_short_word_full_hull_cut_specs(
            lp,
            f_values,
            x_values[num_f : num_f + num_g],
            t_values,
            existing_cut_keys=existing_cut_keys,
            tolerance=tolerance,
            max_words=config.short_word_full_hull_max_words,
            max_word_length=config.short_word_full_hull_max_length,
            max_colors=config.short_word_full_hull_max_colors,
            max_paths=config.word_support_max_paths,
        )
        violations.extend(short_word_full_hull_violations)
    if "short_word_pair_hull" in family_set and (config.run_all_cut_families or not short_word_full_hull_violations):
        violations.extend(
            separate_short_word_pair_hull_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                max_words=config.short_word_pair_hull_max_words,
                max_word_length=config.short_word_pair_hull_max_length,
                max_colors=config.short_word_pair_hull_max_colors,
                max_pair_rows=config.short_word_pair_hull_max_pair_rows,
                max_pairs=config.short_word_pair_hull_max_pairs,
                top_words_per_color=config.short_word_pair_hull_top_words_per_color,
                candidate_word_multiplier=config.short_word_pair_hull_candidate_word_multiplier,
                candidate_top_words_multiplier=config.short_word_pair_hull_candidate_top_words_multiplier,
                candidate_strategy=config.short_word_pair_hull_candidate_strategy,
                candidate_random_seed=config.short_word_pair_hull_candidate_random_seed + 1_000_003 * separation_round,
                pruning=config.short_word_pair_hull_pruning,
                max_paths=config.word_support_max_paths,
                workers=config.short_word_pair_hull_workers,
                batch_size=config.short_word_pair_hull_batch_size,
                min_fractional_shared_colors=config.short_word_pair_hull_min_fractional_shared_colors,
                solution_cache=config.short_word_pair_hull_solution_cache,
                solution_cache_max_entries=config.short_word_pair_hull_cache_max_entries,
                solution_cache_value_quantum=config.short_word_pair_hull_cache_value_quantum,
            )
        )
    if "short_word_pair_single_chain" in family_set:
        violations.extend(
            separate_short_word_pair_chain_template_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                family_name="short_word_pair_single_chain",
                template="single",
                max_words=config.short_word_pair_template_max_words,
                max_word_length=config.short_word_pair_template_max_length,
                candidate_word_multiplier=config.short_word_pair_template_candidate_word_multiplier,
                top_supports_per_shape=config.short_word_pair_template_top_supports_per_shape,
                max_chain_edges=config.short_word_pair_template_max_chain_edges,
                max_template_cuts=config.short_word_pair_template_max_cuts,
                max_paths=config.word_support_max_paths,
            )
        )
    if "short_word_pair_bridge_chain" in family_set:
        violations.extend(
            separate_short_word_pair_chain_template_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                family_name="short_word_pair_bridge_chain",
                template="bridge",
                max_words=config.short_word_pair_template_max_words,
                max_word_length=config.short_word_pair_template_max_length,
                candidate_word_multiplier=config.short_word_pair_template_candidate_word_multiplier,
                top_supports_per_shape=config.short_word_pair_template_top_supports_per_shape,
                max_chain_edges=config.short_word_pair_template_max_chain_edges,
                max_template_cuts=config.short_word_pair_template_max_cuts,
                max_paths=config.word_support_max_paths,
            )
        )
    if "short_word_pair_chains" in family_set:
        violations.extend(
            separate_short_word_pair_chain_template_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                family_name="short_word_pair_chains",
                template="both",
                max_words=config.short_word_pair_template_max_words,
                max_word_length=config.short_word_pair_template_max_length,
                candidate_word_multiplier=config.short_word_pair_template_candidate_word_multiplier,
                top_supports_per_shape=config.short_word_pair_template_top_supports_per_shape,
                max_chain_edges=config.short_word_pair_template_max_chain_edges,
                max_template_cuts=config.short_word_pair_template_max_cuts,
                max_paths=config.word_support_max_paths,
            )
        )
    if "short_word_triple_hull" in family_set and (config.run_all_cut_families or not short_word_full_hull_violations):
        violations.extend(
            separate_short_word_triple_hull_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                max_words=config.short_word_triple_hull_max_words,
                max_word_length=config.short_word_triple_hull_max_length,
                max_rows=config.short_word_triple_hull_max_rows,
                max_triples=config.short_word_triple_hull_max_triples,
                top_words_per_color=config.short_word_triple_hull_top_words_per_color,
                candidate_word_multiplier=config.short_word_triple_hull_candidate_word_multiplier,
                candidate_top_words_multiplier=config.short_word_triple_hull_candidate_top_words_multiplier,
                candidate_sample=config.short_word_triple_hull_candidate_sample,
                candidate_random_seed=config.short_word_triple_hull_candidate_random_seed + 1_000_003 * separation_round,
                token_mode=config.short_word_triple_hull_token_mode,
                max_paths=config.word_support_max_paths,
                workers=config.short_word_triple_hull_workers,
                batch_size=config.short_word_triple_hull_batch_size,
                min_fractional_colors=config.short_word_triple_hull_min_fractional_colors,
            )
        )
    if "short_word_triple_triangle" in family_set:
        violations.extend(
            separate_short_word_triple_template_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                family_name="short_word_triple_triangle",
                template="triangle",
                max_words=config.short_word_triple_template_max_words,
                max_word_length=config.short_word_triple_template_max_length,
                candidate_word_multiplier=config.short_word_triple_template_candidate_word_multiplier,
                top_supports_per_shape=config.short_word_triple_template_top_supports_per_shape,
                max_template_cuts=config.short_word_triple_template_max_cuts,
                validate_templates=config.short_word_triple_template_validate,
            )
        )
    if "short_word_triple_4cycle" in family_set:
        violations.extend(
            separate_short_word_triple_template_cut_specs(
                lp,
                f_values,
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                family_name="short_word_triple_4cycle",
                template="4cycle",
                max_words=config.short_word_triple_template_max_words,
                max_word_length=config.short_word_triple_template_max_length,
                candidate_word_multiplier=config.short_word_triple_template_candidate_word_multiplier,
                top_supports_per_shape=config.short_word_triple_template_top_supports_per_shape,
                max_template_cuts=config.short_word_triple_template_max_cuts,
                validate_templates=config.short_word_triple_template_validate,
            )
        )
    if "group_value_deep" in family_set:
        violations.extend(
            separate_group_value_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                seed_words=250,
                candidate_words=3000,
                group_words=500,
                max_rank=10,
            )
        )
    if "conflict_clique" in family_set:
        violations.extend(
            separate_conflict_clique_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "conflict_odd_cycle" in family_set:
        violations.extend(
            separate_conflict_odd_cycle_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "word_support" in family_set:
        violations.extend(
            separate_word_support_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
                max_words=config.word_support_max_words,
                max_rank=config.word_support_max_rank,
                max_paths=config.word_support_max_paths,
            )
        )
    if "bad_vocab_escape" in family_set:
        violations.extend(
            separate_bad_vocab_escape_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
            )
        )
    if "bad_vocab_improvement" in family_set:
        violations.extend(
            separate_bad_vocab_improvement_cut_specs(
                lp,
                f_values,
                x_values[num_f : num_f + num_g],
                t_values,
                existing_cut_keys=existing_cut_keys,
                tolerance=tolerance,
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


class PerFamilyCutList(list):
    def __init__(self, *, max_per_extend: int):
        super().__init__()
        self.max_per_extend = max(0, int(max_per_extend))

    def extend(self, cuts):
        rows = list(cuts)
        if self.max_per_extend > 0 and len(rows) > self.max_per_extend:
            rows.sort(key=lambda item: item[0], reverse=True)
            LOGGER.info(
                "Limiting one cut family from %d candidate cuts to top %d",
                len(rows),
                self.max_per_extend,
            )
            rows = rows[: self.max_per_extend]
        super().extend(rows)


def separate_word_rank_cut_specs(
    lp,
    f_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    family_name: str,
    edge_weight: str,
    max_words: int = 500,
    max_rank: int = 5,
):
    """Separate full-word small-rank interval-capacity cuts."""

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    for word_idx in rank_suspicious_words(lp, f_values, max_words=max_words):
        by_color = defaultdict(list)
        scores = defaultdict(float)
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            value = float(f_values[edge_idx])
            if value <= tolerance:
                continue
            info = lp["nonfree_edge_info"][edge_idx]
            coeff = edge_capacity_weight(info, edge_weight)
            token_idx = info["token_index"]
            by_color[token_idx].append((edge_idx, coeff))
            scores[token_idx] += coeff * value

        if len(scores) < 2:
            continue
        selected_tokens = tuple(
            token_idx
            for token_idx, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_rank]
        )
        full_key = (family_name, word_idx, selected_tokens)
        if full_key in existing_cut_keys:
            continue

        selected_edges = [
            (edge_idx, token_idx, coeff)
            for token_idx in selected_tokens
            for edge_idx, coeff in by_color[token_idx]
        ]
        capacities = subset_edge_capacities(lp, selected_edges, selected_tokens)
        bound = modular_upper_bound_for_capacities(capacities, t_values, selected_tokens)
        if bound is None:
            continue
        alpha, beta_values, rhs_at_current = bound
        usage = sum(coeff * float(f_values[edge_idx]) for edge_idx, _, coeff in selected_edges)
        violation = usage - rhs_at_current
        if violation <= tolerance:
            continue

        beta_key = tuple(round(float(beta), 8) for beta in beta_values)
        keyed_full_key = (*full_key, beta_key)
        if keyed_full_key in existing_cut_keys:
            continue

        entries = [(edge_idx, float(coeff)) for edge_idx, _, coeff in selected_edges]
        entries.extend(
            (t_offset + token_idx, -float(beta))
            for token_idx, beta in zip(selected_tokens, beta_values)
            if abs(beta) > 1e-12
        )
        violations.append((violation, keyed_full_key, entries, float(alpha)))

    return violations


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


def separate_global_rank_count_cut_specs(
    lp,
    f_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    rank: int = 3,
    max_words: int = 1000,
    top_colors_per_word: int = 8,
):
    """Separate weighted corpus-level small-rank count-capacity cuts."""

    rank_data = {}
    for word_idx in rank_suspicious_words(lp, f_values, max_words=max_words):
        by_color = defaultdict(list)
        scores = defaultdict(float)
        word_weight = float(lp["word_weights"][word_idx])
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            value = float(f_values[edge_idx])
            if value <= tolerance:
                continue
            token_idx = lp["nonfree_edge_info"][edge_idx]["token_index"]
            by_color[token_idx].append(edge_idx)
            scores[token_idx] += word_weight * value

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
                data["entries"].extend((edge_idx, word_weight) for edge_idx in by_color[token_idx])

            for mask in range(1, 1 << rank):
                mask_edges = []
                for bit, token_idx in enumerate(color_tuple):
                    if mask & (1 << bit):
                        mask_edges.extend(by_color[token_idx])
                data["capacities"][mask] += word_weight * color_window_capacity(
                    lp,
                    [(edge_idx, 1.0) for edge_idx in mask_edges],
                )

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    for color_tuple, data in rank_data.items():
        full_key = ("global_rank_count", *color_tuple)
        if full_key in existing_cut_keys:
            continue
        bound = modular_upper_bound_for_capacities(data["capacities"], t_values, color_tuple)
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


def rank_short_fractional_words(
    lp,
    f_values,
    t_values,
    *,
    max_words: int,
    max_word_length: int,
    tolerance: float,
):
    rows = []
    for word_idx, edge_indices in lp["word_nonfree_edges"].items():
        word_len = lp["word_lengths"][word_idx]
        if word_len > max_word_length:
            continue
        score = 0.0
        for edge_idx in edge_indices:
            edge_value = float(f_values[edge_idx])
            if not (tolerance < edge_value < 1.0 - tolerance):
                continue
            info = lp["nonfree_edge_info"][edge_idx]
            token_value = float(t_values[info["token_index"]])
            if not (tolerance < token_value < 1.0 - tolerance):
                continue
            score += (
                min(edge_value, 1.0 - edge_value)
                * min(token_value, 1.0 - token_value)
                * max(1, info["end"] - info["start"])
            )
        if score <= 0.0:
            continue
        weight = float(lp["word_weights"][word_idx])
        rows.append((-weight * score, word_len, -weight, word_idx))
    rows.sort()
    return [word_idx for _, _, _, word_idx in rows[:max_words]]


def rank_frequent_short_fractional_words(
    lp,
    f_values,
    t_values,
    *,
    max_words: int,
    max_word_length: int,
    tolerance: float,
):
    rows = []
    for word_idx, edge_indices in lp["word_nonfree_edges"].items():
        word_len = lp["word_lengths"][word_idx]
        if word_len > max_word_length:
            continue
        fractional_signal = 0.0
        for edge_idx in edge_indices:
            edge_value = float(f_values[edge_idx])
            info = lp["nonfree_edge_info"][edge_idx]
            token_value = float(t_values[info["token_index"]])
            if tolerance < edge_value < 1.0 - tolerance or tolerance < token_value < 1.0 - tolerance:
                fractional_signal += (
                    min(max(edge_value, token_value), 1.0 - min(edge_value, token_value))
                    * max(1, info["end"] - info["start"])
                )
        if fractional_signal <= 0.0:
            continue
        weight = float(lp["word_weights"][word_idx])
        rows.append((-weight, word_len, -weight * fractional_signal, word_idx))
    rows.sort()
    return [word_idx for _, _, _, word_idx in rows[:max_words]]


def all_word_token_colors(lp, word_idx: int):
    return tuple(
        sorted(
            {
                lp["nonfree_edge_info"][edge_idx]["token_index"]
                for edge_idx in lp["word_nonfree_edges"].get(word_idx, [])
            }
        )
    )


def short_word_pair_candidates(
    lp,
    f_values,
    t_values,
    *,
    max_words: int,
    max_word_length: int,
    top_words_per_color: int,
    tolerance: float,
):
    ranked_words = rank_short_fractional_words(
        lp,
        f_values,
        t_values,
        max_words=max_words,
        max_word_length=max_word_length,
        tolerance=tolerance,
    )
    word_color_scores = {}
    color_to_words = defaultdict(list)
    for word_idx in ranked_words:
        scores = defaultdict(float)
        word_weight = float(lp["word_weights"][word_idx])
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            edge_value = float(f_values[edge_idx])
            info = lp["nonfree_edge_info"][edge_idx]
            token_idx = info["token_index"]
            token_value = float(t_values[token_idx])
            if tolerance < edge_value < 1.0 - tolerance and tolerance < token_value < 1.0 - tolerance:
                scores[token_idx] += min(edge_value, 1.0 - edge_value) * max(1, info["end"] - info["start"])
        word_color_scores[word_idx] = scores
        for token_idx, score in scores.items():
            color_to_words[token_idx].append((word_weight * score, word_idx))

    for rows in color_to_words.values():
        rows.sort(reverse=True)

    pair_scores = defaultdict(float)
    for token_idx, rows in color_to_words.items():
        token_value = float(t_values[token_idx])
        for (_, left_word), (_, right_word) in combinations(rows[:top_words_per_color], 2):
            if left_word == right_word:
                continue
            key = tuple(sorted((left_word, right_word)))
            pair_scores[key] += (
                min(
                    word_color_scores[left_word].get(token_idx, 0.0),
                    word_color_scores[right_word].get(token_idx, 0.0),
                )
                * token_value
            )

    return [
        (score, left_word, right_word)
        for (left_word, right_word), score in sorted(
            pair_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ], word_color_scores


def reorder_short_word_pair_candidates(
    pair_rows,
    lp,
    t_values,
    *,
    max_pairs: int,
    tolerance: float,
    strategy: str,
    random_seed: int,
):
    if strategy == "score":
        return pair_rows
    if strategy == "random":
        shuffled = list(pair_rows)
        rng = random.Random(random_seed)
        rng.shuffle(shuffled)
        return shuffled
    if strategy != "mixed":
        raise ValueError(f"Unsupported short_word_pair_hull candidate strategy {strategy!r}")

    fractional_colors = set(np.flatnonzero((t_values > tolerance) & (t_values < 1.0 - tolerance)))
    colors_by_word = {}
    enriched = []
    for score, left_word, right_word in pair_rows:
        left_colors = colors_by_word.setdefault(left_word, set(all_word_token_colors(lp, left_word)))
        right_colors = colors_by_word.setdefault(right_word, set(all_word_token_colors(lp, right_word)))
        shared_fractional_count = len((left_colors & right_colors) & fractional_colors)
        fractional_count = len((left_colors | right_colors) & fractional_colors)
        enriched.append(
            {
                "row": (score, left_word, right_word),
                "score": score,
                "shared_fractional_count": shared_fractional_count,
                "fractional_count": fractional_count,
            }
        )

    quota = max(1, int(math.ceil(max_pairs / 4)))
    selected = []
    selected_keys = set()

    def add_rows(rows):
        for row in rows:
            _, left_word, right_word = row["row"]
            key = (left_word, right_word)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(row["row"])

    add_rows(
        sorted(
            enriched,
            key=lambda item: (item["score"], item["shared_fractional_count"], item["fractional_count"]),
            reverse=True,
        )[:quota]
    )
    add_rows(
        sorted(
            enriched,
            key=lambda item: (item["shared_fractional_count"], item["score"], item["fractional_count"]),
            reverse=True,
        )[:quota]
    )
    add_rows(
        sorted(
            enriched,
            key=lambda item: (item["fractional_count"], item["score"], item["shared_fractional_count"]),
            reverse=True,
        )[:quota]
    )

    remaining = [row["row"] for row in enriched if (row["row"][1], row["row"][2]) not in selected_keys]
    rng = random.Random(random_seed)
    rng.shuffle(remaining)
    selected.extend(remaining)
    return selected


def resolve_pair_hull_workers(workers: int) -> int:
    if workers > 0:
        return workers
    return max(1, (os.cpu_count() or 2) - 1)


PAIR_HULL_WORKER_STATE = {}


def init_pair_hull_worker(state):
    global PAIR_HULL_WORKER_STATE
    PAIR_HULL_WORKER_STATE = state


def pair_hull_worker(task):
    if len(task) == 4:
        left_word, right_word, selected_tokens, cache_key = task
    else:
        left_word, right_word, selected_tokens = task
        cache_key = None
    state = PAIR_HULL_WORKER_STATE
    lp = state["lp"]
    left_paths = state["paths_by_word"][left_word]
    right_paths = state["paths_by_word"][right_word]
    if state.get("pruning", "full") == "fractional_edges_shared_colors":
        return reduced_pair_hull_worker_result(
            lp,
            state["f_values"],
            state["g_values"],
            state["t_values"],
            left_word,
            right_word,
            selected_tokens,
            left_paths,
            right_paths,
            tolerance=state["tolerance"],
            max_pair_rows=state["max_pair_rows"],
            cache_key=cache_key,
        )
    cut = pair_full_upward_hull_cut_from_paths(
        lp,
        state["f_values"],
        state["g_values"],
        state["t_values"],
        left_word,
        right_word,
        selected_tokens,
        left_paths,
        right_paths,
        tolerance=state["tolerance"],
    )
    if cut is None:
        return {"cut": None, "build_seconds": 0.0, "solve_seconds": 0.0, "cache_key": cache_key}

    violation, edge_coefficients, token_coefficients, rhs, coefficient_key, build_seconds, solve_seconds = cut
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g
    entries = []
    entries.extend((col_idx, coefficient) for col_idx, coefficient in edge_coefficients.items())
    entries.extend(
        (t_offset + token_idx, coefficient)
        for token_idx, coefficient in token_coefficients.items()
    )
    full_key = ("short_word_pair_hull", left_word, right_word, selected_tokens, coefficient_key)
    return {
        "cut": (violation, full_key, entries, rhs),
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "validation_seconds": 0.0,
        "cache_key": cache_key,
    }


def reduced_pair_hull_worker_result(
    lp,
    f_values,
    g_values,
    t_values,
    left_word,
    right_word,
    selected_tokens,
    left_paths,
    right_paths,
    *,
    tolerance,
    max_pair_rows,
    cache_key,
):
    cut = pair_reduced_fractional_edge_hull_cut_from_paths(
        lp,
        f_values,
        g_values,
        t_values,
        left_word,
        right_word,
        selected_tokens,
        left_paths,
        right_paths,
        tolerance=tolerance,
        max_pair_rows=max_pair_rows,
    )
    base = {
        "cut": None,
        "build_seconds": 0.0,
        "solve_seconds": 0.0,
        "validation_seconds": 0.0,
        "cache_key": cache_key,
        "skip_reason": None,
        "reduced_rows": 0,
        "edge_vars": 0,
    }
    if cut is None:
        return base
    if cut.get("skip_reason") is not None:
        base.update(cut)
        return base
    if "edge_coefficients" not in cut:
        base.update(cut)
        return base

    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g
    entries = []
    entries.extend((col_idx, coefficient) for col_idx, coefficient in cut["edge_coefficients"].items())
    entries.extend(
        (t_offset + token_idx, coefficient)
        for token_idx, coefficient in cut["token_coefficients"].items()
    )
    full_key = (
        "short_word_pair_hull",
        left_word,
        right_word,
        selected_tokens,
        cut["coefficient_key"],
    )
    base.update(cut)
    base["cut"] = (cut["violation"], full_key, entries, cut["rhs"])
    return base


def pair_hull_batch_worker(tasks):
    return [pair_hull_worker(task) for task in tasks]


TRIPLE_HULL_WORKER_STATE = {}


def init_triple_hull_worker(state):
    global TRIPLE_HULL_WORKER_STATE
    TRIPLE_HULL_WORKER_STATE = state


def triple_hull_worker(task):
    left_word, middle_word, right_word, selected_tokens = task
    state = TRIPLE_HULL_WORKER_STATE
    lp = state["lp"]
    cut = triple_reduced_fractional_edge_hull_cut_from_paths(
        lp,
        state["f_values"],
        state["g_values"],
        state["t_values"],
        (left_word, middle_word, right_word),
        selected_tokens,
        [state["paths_by_word"][word_idx] for word_idx in (left_word, middle_word, right_word)],
        tolerance=state["tolerance"],
        max_rows=state["max_rows"],
    )
    base = {
        "cut": None,
        "build_seconds": 0.0,
        "solve_seconds": 0.0,
        "validation_seconds": 0.0,
        "skip_reason": None,
        "reduced_rows": 0,
        "edge_vars": 0,
    }
    if cut is None:
        return base
    if cut.get("skip_reason") is not None:
        base.update(cut)
        return base
    if "edge_coefficients" not in cut:
        base.update(cut)
        return base

    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g
    entries = []
    entries.extend((col_idx, coefficient) for col_idx, coefficient in cut["edge_coefficients"].items())
    entries.extend(
        (t_offset + token_idx, coefficient)
        for token_idx, coefficient in cut["token_coefficients"].items()
    )
    full_key = (
        "short_word_triple_hull",
        left_word,
        middle_word,
        right_word,
        selected_tokens,
        cut["coefficient_key"],
    )
    base.update(cut)
    base["cut"] = (cut["violation"], full_key, entries, cut["rhs"])
    return base


def triple_hull_batch_worker(tasks):
    return [triple_hull_worker(task) for task in tasks]


def pair_hull_projection_cache_key(
    lp,
    f_values,
    g_values,
    t_values,
    left_word: int,
    right_word: int,
    selected_tokens,
    *,
    pruning: str,
    value_quantum: float,
):
    """Key a pair-hull separator LP by structure and rounded projected LP values."""

    num_f = lp["num_nonfree_edges"]
    projected_values = []
    for word_idx in (left_word, right_word):
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            projected_values.append(float(f_values[edge_idx]))
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            projected_values.append(float(g_values[edge_idx]))
    token_values = [float(t_values[token_idx]) for token_idx in selected_tokens]

    hasher = hashlib.sha256()
    hasher.update(b"tokenizer-lp-pair-hull-projection-v2")
    hasher.update(str(pruning).encode("utf-8"))
    hasher.update(f"value_quantum={float(value_quantum):.17g}".encode("ascii"))
    hash_array(hasher, np.asarray([left_word, right_word, num_f], dtype=np.int64))
    hash_array(hasher, np.asarray(selected_tokens, dtype=np.int64))
    hash_projected_values(hasher, projected_values, value_quantum=value_quantum)
    hash_projected_values(hasher, token_values, value_quantum=value_quantum)
    return hasher.hexdigest()


def hash_projected_values(hasher, values, *, value_quantum: float) -> None:
    values_array = np.asarray(values, dtype=np.float64)
    if value_quantum > 0:
        quantized = np.rint(values_array / float(value_quantum)).astype(np.int64)
        hash_array(hasher, quantized)
    else:
        hash_array(hasher, values_array)


def cut_violation_from_entries(lp, f_values, g_values, t_values, entries, rhs: float) -> float:
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g
    lhs = 0.0
    for col_idx, coefficient in entries:
        if col_idx < num_f:
            value = f_values[col_idx]
        elif col_idx < t_offset:
            value = g_values[col_idx - num_f]
        else:
            value = t_values[col_idx - t_offset]
        lhs += float(coefficient) * float(value)
    return lhs - float(rhs)


def maybe_cache_pair_hull_result(
    cache: dict[str, dict | None] | None,
    cache_key: str | None,
    result,
    max_entries: int,
) -> None:
    if cache is None or cache_key is None or max_entries <= 0 or cache_key in cache or len(cache) >= max_entries:
        return
    cache[cache_key] = result


def chunked(items, chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def edge_capacity_weight(info, edge_weight: str) -> float:
    if edge_weight == "count":
        return 1.0
    if edge_weight == "length":
        return float(max(1, info["end"] - info["start"]))
    raise ValueError(f"Unsupported edge capacity weight {edge_weight!r}")


def subset_window_capacities(lp, selected_edges, selected_tokens):
    return subset_edge_capacities(lp, selected_edges, selected_tokens)


def subset_edge_capacities(lp, selected_edges, selected_tokens):
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


def intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


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


def separate_path_min_cover_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 150,
    max_paths_per_word: int = 20000,
    max_universe: int = 80,
):
    """Separate path-cover cuts with an exact minimum-weight hitting set."""

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

        token_universe = sorted(set().union(*path_token_sets))
        if len(token_universe) > max_universe:
            continue
        hitting_set = exact_min_weight_hitting_set(path_token_sets, t_values, token_universe)
        if not hitting_set:
            continue

        full_key = ("path_min_cover", word_idx, k_value, tuple(sorted(hitting_set)))
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

    return violations


def separate_group_value_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    seed_words: int = 120,
    candidate_words: int = 1200,
    group_words: int = 250,
    max_rank: int = 8,
):
    """Separate small token-set lower-envelope cuts over groups of words.

    For a selected token set S and a group of words G, define F(U) as the
    minimum weighted token count on G when U subset S is active and every token
    outside S is allowed. Any affine lower bound on F(U) is valid for integral
    token activations, and can cut fractional solutions that combine
    word-specific alternatives inconsistently.
    """

    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g
    suspicious = rank_suspicious_words(lp, f_values, max_words=candidate_words)
    suspicious_set = set(suspicious)
    token_word_scores = defaultdict(list)
    word_fractional_scores = {}

    for word_idx in suspicious:
        score = 0.0
        by_token_score = defaultdict(float)
        word_weight = float(lp["word_weights"][word_idx])
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            value = float(f_values[edge_idx])
            if value <= tolerance:
                continue
            info = lp["nonfree_edge_info"][edge_idx]
            token_idx = info["token_index"]
            edge_score = value * max(1, info["end"] - info["start"])
            by_token_score[token_idx] += edge_score
            if tolerance < float(t_values[token_idx]) < 1.0 - tolerance:
                score += min(value, 1.0 - value) * max(1, info["end"] - info["start"])
        if score <= 0:
            continue
        word_fractional_scores[word_idx] = word_weight * score
        for token_idx, token_score in by_token_score.items():
            token_word_scores[token_idx].append((word_weight * token_score, word_idx))

    for rows in token_word_scores.values():
        rows.sort(reverse=True)

    candidate_sets = []
    for word_idx in suspicious[:seed_words]:
        color_scores = defaultdict(float)
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            value = float(f_values[edge_idx])
            if value <= tolerance:
                continue
            info = lp["nonfree_edge_info"][edge_idx]
            token_idx = info["token_index"]
            token_value = float(t_values[token_idx])
            if tolerance < token_value < 1.0 - tolerance:
                color_scores[token_idx] += value * max(1, info["end"] - info["start"])
        selected = tuple(
            token_idx
            for token_idx, _ in sorted(color_scores.items(), key=lambda item: item[1], reverse=True)[:max_rank]
        )
        if len(selected) >= 2:
            candidate_sets.append(selected)

    seen_sets = set()
    violations = []
    for selected_tokens in candidate_sets:
        selected_tokens = tuple(sorted(selected_tokens))
        if selected_tokens in seen_sets:
            continue
        seen_sets.add(selected_tokens)
        full_key_prefix = ("group_value", selected_tokens)
        if any(key[:2] == full_key_prefix for key in existing_cut_keys):
            continue

        group_score = defaultdict(float)
        for token_idx in selected_tokens:
            for score, word_idx in token_word_scores.get(token_idx, [])[:group_words]:
                if word_idx in suspicious_set:
                    group_score[word_idx] += score
        selected_words = tuple(
            word_idx
            for word_idx, _ in sorted(
                group_score.items(),
                key=lambda item: (item[1], word_fractional_scores.get(item[0], 0.0)),
                reverse=True,
            )[:group_words]
        )
        if not selected_words:
            continue

        current_cost = group_current_cost(lp, selected_words, f_values, g_values)
        value_by_mask = group_value_function(lp, selected_words, selected_tokens)
        bound = affine_lower_bound_for_values(value_by_mask, t_values, selected_tokens)
        if bound is None:
            continue
        alpha, beta_values, rhs_at_current = bound
        violation = rhs_at_current - current_cost
        if violation <= tolerance:
            continue

        beta_key = tuple(round(float(beta), 8) for beta in beta_values)
        full_key = (*full_key_prefix, selected_words, beta_key)
        if full_key in existing_cut_keys:
            continue

        entries = []
        for word_idx in selected_words:
            word_weight = float(lp["word_weights"][word_idx])
            entries.extend((edge_idx, -word_weight) for edge_idx in lp["word_nonfree_edges"].get(word_idx, []))
            entries.extend((num_f + edge_idx, -word_weight) for edge_idx in lp["word_free_edges"].get(word_idx, []))
        entries.extend(
            (t_offset + token_idx, float(beta))
            for token_idx, beta in zip(selected_tokens, beta_values)
            if abs(beta) > 1e-12
        )
        violations.append((violation, full_key, entries, -float(alpha)))

    return violations


def separate_threshold_value_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 1000,
    max_rank: int = 12,
):
    """Separate per-word value-function cuts over selected token colours.

    For a word w and selected colours S, define F(U) as the shortest
    segmentation cost when U subset S is active and every colour outside S is
    left unrestricted. Any affine lower bound on F(U) is valid for the full
    ILP, because omitted colours can only make F smaller.
    """

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    for word_idx in rank_suspicious_words(lp, f_values, max_words=max_words):
        selected_tokens = word_support_selected_tokens(
            lp,
            f_values,
            t_values,
            word_idx,
            max_rank=max_rank,
            tolerance=tolerance,
        )
        if len(selected_tokens) < 2:
            continue
        full_key_prefix = ("threshold_value", word_idx, selected_tokens)
        if any(key[:3] == full_key_prefix for key in existing_cut_keys):
            continue

        word_weight = float(lp["word_weights"][word_idx])
        current_cost = word_weight * float(
            f_values[lp["word_nonfree_edges"].get(word_idx, [])].sum()
            + g_values[lp["word_free_edges"].get(word_idx, [])].sum()
        )
        value_by_mask = group_value_function(lp, (word_idx,), selected_tokens)
        bound = affine_lower_bound_for_values(value_by_mask, t_values, selected_tokens)
        if bound is None:
            continue
        alpha, beta_values, rhs_at_current = bound
        violation = rhs_at_current - current_cost
        if violation <= tolerance:
            continue

        beta_key = tuple(round(float(beta), 8) for beta in beta_values)
        full_key = (*full_key_prefix, beta_key)
        if full_key in existing_cut_keys:
            continue

        entries = []
        entries.extend((edge_idx, -word_weight) for edge_idx in lp["word_nonfree_edges"].get(word_idx, []))
        entries.extend((num_f + edge_idx, -word_weight) for edge_idx in lp["word_free_edges"].get(word_idx, []))
        entries.extend(
            (t_offset + token_idx, float(beta))
            for token_idx, beta in zip(selected_tokens, beta_values)
            if abs(beta) > 1e-12
        )
        violations.append((violation, full_key, entries, -float(alpha)))

    return violations


def separate_group_budget_value_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    seed_words: int = 120,
    candidate_words: int = 1500,
    group_words: int = 300,
    max_rank: int = 10,
):
    """Separate group cuts depending only on local selected-vocab size.

    For a word group G and token set S, B(k) is the best weighted segmentation
    cost obtainable with at most k active colours from S, with all colours
    outside S unrestricted. A lower affine support of B(sum_S t) is ILP-valid.
    """

    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g
    suspicious = rank_suspicious_words(lp, f_values, max_words=candidate_words)
    suspicious_set = set(suspicious)
    token_word_scores = defaultdict(list)
    word_fractional_scores = {}

    for word_idx in suspicious:
        score = 0.0
        by_token_score = defaultdict(float)
        word_weight = float(lp["word_weights"][word_idx])
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            value = float(f_values[edge_idx])
            if value <= tolerance:
                continue
            info = lp["nonfree_edge_info"][edge_idx]
            token_idx = info["token_index"]
            edge_score = value * max(1, info["end"] - info["start"])
            by_token_score[token_idx] += edge_score
            if tolerance < float(t_values[token_idx]) < 1.0 - tolerance:
                score += min(value, 1.0 - value) * max(1, info["end"] - info["start"])
        if score <= 0:
            continue
        word_fractional_scores[word_idx] = word_weight * score
        for token_idx, token_score in by_token_score.items():
            token_word_scores[token_idx].append((word_weight * token_score, word_idx))

    for rows in token_word_scores.values():
        rows.sort(reverse=True)

    candidate_sets = []
    for word_idx in suspicious[:seed_words]:
        selected = word_support_selected_tokens(
            lp,
            f_values,
            t_values,
            word_idx,
            max_rank=max_rank,
            tolerance=tolerance,
        )
        if len(selected) >= 2:
            candidate_sets.append(selected)

    seen_sets = set()
    violations = []
    for selected_tokens in candidate_sets:
        selected_tokens = tuple(sorted(selected_tokens))
        if selected_tokens in seen_sets:
            continue
        seen_sets.add(selected_tokens)
        full_key_prefix = ("group_budget_value", selected_tokens)
        if any(key[:2] == full_key_prefix for key in existing_cut_keys):
            continue

        group_score = defaultdict(float)
        for token_idx in selected_tokens:
            for score, word_idx in token_word_scores.get(token_idx, [])[:group_words]:
                if word_idx in suspicious_set:
                    group_score[word_idx] += score
        selected_words = tuple(
            word_idx
            for word_idx, _ in sorted(
                group_score.items(),
                key=lambda item: (item[1], word_fractional_scores.get(item[0], 0.0)),
                reverse=True,
            )[:group_words]
        )
        if not selected_words:
            continue

        current_cost = group_current_cost(lp, selected_words, f_values, g_values)
        values_by_budget = group_budget_value_function(lp, selected_words, selected_tokens)
        bound = affine_lower_bound_for_budget_values(values_by_budget, t_values, selected_tokens)
        if bound is None:
            continue
        alpha, beta, rhs_at_current = bound
        violation = rhs_at_current - current_cost
        if violation <= tolerance:
            continue

        full_key = (*full_key_prefix, selected_words, round(float(alpha), 8), round(float(beta), 8))
        if full_key in existing_cut_keys:
            continue

        entries = []
        for word_idx in selected_words:
            word_weight = float(lp["word_weights"][word_idx])
            entries.extend((edge_idx, -word_weight) for edge_idx in lp["word_nonfree_edges"].get(word_idx, []))
            entries.extend((num_f + edge_idx, -word_weight) for edge_idx in lp["word_free_edges"].get(word_idx, []))
        entries.extend(
            (t_offset + token_idx, float(beta))
            for token_idx in selected_tokens
            if abs(beta) > 1e-12
        )
        violations.append((violation, full_key, entries, -float(alpha)))

    return violations


def separate_conflict_clique_cut_specs(
    lp,
    f_values,
    g_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 2000,
):
    """Separate interval-conflict clique cuts within words.

    A tokenizer path can use at most one interval edge crossing any byte
    boundary. These cuts are valid but should usually be implied by the word
    flow conservation equations; the separator is useful as a sanity check.
    """

    violations = []
    num_f = lp["num_nonfree_edges"]
    suspicious_words = rank_suspicious_words(lp, f_values, max_words=max_words)
    if not suspicious_words:
        suspicious_words = range(len(lp["word_lengths"]))

    for word_idx in suspicious_words:
        word_len = lp["word_lengths"][word_idx]
        for boundary in range(word_len):
            full_key = ("conflict_clique", word_idx, boundary)
            if full_key in existing_cut_keys:
                continue
            entries = []
            lhs_value = 0.0
            for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
                info = lp["nonfree_edge_info"][edge_idx]
                if info["start"] <= boundary < info["end"]:
                    entries.append((edge_idx, 1.0))
                    lhs_value += float(f_values[edge_idx])
            for edge_idx in lp["word_free_edges"].get(word_idx, []):
                info = lp["free_edge_info"][edge_idx]
                if info["start"] <= boundary < info["end"]:
                    entries.append((num_f + edge_idx, 1.0))
                    lhs_value += float(g_values[edge_idx])
            violation = lhs_value - 1.0
            if violation > tolerance:
                violations.append((violation, full_key, entries, 1.0))

    return violations


def separate_conflict_odd_cycle_cut_specs(
    lp,
    f_values,
    g_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 500,
    max_vertices: int = 80,
    max_cycle_len: int = 11,
):
    """Separate odd-cycle cuts in the per-word interval conflict graph.

    Vertices are concrete token/free-byte interval edges in a single word. Two
    vertices conflict if their spans overlap. In any integral segmentation, the
    chosen intervals form a stable set, so for every odd cycle C:

        sum_{e in C} x_e <= floor(|C| / 2)
    """

    violations = []
    for word_idx in rank_suspicious_words(lp, f_values, max_words=max_words):
        vertices = conflict_vertices_for_word(
            lp,
            f_values,
            g_values,
            word_idx,
            max_vertices=max_vertices,
            tolerance=tolerance,
        )
        if len(vertices) < 5:
            continue
        adjacency = build_interval_conflict_adjacency(vertices)
        for cycle in find_odd_cycles(adjacency, max_cycle_len=max_cycle_len):
            columns = tuple(sorted(vertices[idx]["col"] for idx in cycle))
            full_key = ("conflict_odd_cycle", word_idx, columns)
            if full_key in existing_cut_keys:
                continue
            lhs_value = sum(vertices[idx]["value"] for idx in cycle)
            rhs_value = len(cycle) // 2
            violation = lhs_value - rhs_value
            if violation <= tolerance:
                continue
            entries = [(vertices[idx]["col"], 1.0) for idx in cycle]
            violations.append((violation, full_key, entries, float(rhs_value)))

    return violations


def separate_word_support_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 2000,
    max_rank: int = 20,
    max_paths: int = 100000,
):
    """Separate exact word segmentation support cuts.

    For a word and selected token colours S, every segmentation path p satisfies:

        dual_edge_sum(p) + gamma <= |R(p) cap S|

    Since all colours in R(p) must be active in any integral tokenizer using p,
    the projected inequality

        dual_edge_sum(flow_w) + gamma <= sum_{tau in S} t_tau

    is ILP-valid. We only emit cuts when all segmentations of the word were
    enumerated, so the dual constraints cover the full path set.
    """

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    for word_idx in rank_suspicious_words(lp, f_values, max_words=max_words):
        if lp["word_weights"][word_idx] <= 1:
            continue
        selected_tokens = word_support_selected_tokens(
            lp,
            f_values,
            t_values,
            word_idx,
            max_rank=max_rank,
            tolerance=tolerance,
        )
        if len(selected_tokens) < 2:
            continue
        full_key_prefix = ("word_support", word_idx, selected_tokens)
        if any(key[:3] == full_key_prefix for key in existing_cut_keys):
            continue

        paths = enumerate_word_edge_paths(lp, word_idx, max_paths=max_paths)
        if paths is None or len(paths) < 2:
            continue
        cut = word_support_cut_from_paths(
            lp,
            f_values,
            g_values,
            t_values,
            word_idx,
            selected_tokens,
            paths,
            tolerance=tolerance,
        )
        if cut is None:
            continue
        violation, edge_coefficients, token_coefficients, rhs, gamma = cut
        gamma_key = round(float(gamma), 8)
        full_key = (*full_key_prefix, gamma_key)
        if full_key in existing_cut_keys:
            continue

        entries = []
        entries.extend((col_idx, coefficient) for col_idx, coefficient in edge_coefficients.items())
        entries.extend(
            (t_offset + token_idx, coefficient)
            for token_idx, coefficient in token_coefficients.items()
        )
        violations.append((violation, full_key, entries, rhs))

    return violations


def separate_word_hull_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 1000,
    max_rank: int = 16,
    max_paths: int = 100000,
):
    """Separate weighted exact path-hull support cuts for one word.

    For selected colours S, enumerate all paths p through a word. The separator
    searches for nonnegative token coefficients b, sum(b)<=1, and edge
    potentials y,gamma such that:

        y·incidence(p) + gamma <= sum_{tau in R(p) cap S} b_tau

    This implies y·flow + gamma <= b·t for every integral tokenizer. Optimizing
    b at the current fractional solution gives a small exact support-hull cut.
    """

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    for word_idx in rank_suspicious_words(lp, f_values, max_words=max_words):
        if lp["word_weights"][word_idx] <= 1:
            continue
        selected_tokens = word_support_selected_tokens(
            lp,
            f_values,
            t_values,
            word_idx,
            max_rank=max_rank,
            tolerance=tolerance,
        )
        if len(selected_tokens) < 2:
            continue
        full_key_prefix = ("word_hull", word_idx, selected_tokens)
        if any(key[:3] == full_key_prefix for key in existing_cut_keys):
            continue

        paths = enumerate_word_edge_paths(lp, word_idx, max_paths=max_paths)
        if paths is None or len(paths) < 2:
            continue
        cut = word_hull_cut_from_paths(
            lp,
            f_values,
            g_values,
            t_values,
            word_idx,
            selected_tokens,
            paths,
            tolerance=tolerance,
        )
        if cut is None:
            continue
        violation, edge_coefficients, token_coefficients, rhs, coefficient_key = cut
        full_key = (*full_key_prefix, coefficient_key)
        if full_key in existing_cut_keys:
            continue

        entries = []
        entries.extend((col_idx, coefficient) for col_idx, coefficient in edge_coefficients.items())
        entries.extend(
            (t_offset + token_idx, coefficient)
            for token_idx, coefficient in token_coefficients.items()
        )
        violations.append((violation, full_key, entries, rhs))

    return violations


def separate_short_word_hull_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 2000,
    max_word_length: int = 10,
    max_rank: int = 16,
    max_paths: int = 100000,
):
    """Separate exact projected hull cuts on short words with fractional colours."""

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g
    path_pattern_cache = {}

    for word_idx in rank_short_fractional_words(
        lp,
        f_values,
        t_values,
        max_words=max_words,
        max_word_length=max_word_length,
        tolerance=tolerance,
    ):
        selected_tokens = word_support_selected_tokens(
            lp,
            f_values,
            t_values,
            word_idx,
            max_rank=max_rank,
            tolerance=tolerance,
        )
        if len(selected_tokens) < 2:
            continue
        full_key_prefix = ("short_word_hull", word_idx, selected_tokens)
        if any(key[:3] == full_key_prefix for key in existing_cut_keys):
            continue

        paths = enumerate_word_edge_paths_by_pattern(
            lp,
            word_idx,
            max_paths=max_paths,
            cache=path_pattern_cache,
        )
        if paths is None or len(paths) < 2:
            continue
        cut = word_hull_cut_from_paths(
            lp,
            f_values,
            g_values,
            t_values,
            word_idx,
            selected_tokens,
            paths,
            tolerance=tolerance,
        )
        if cut is None:
            continue
        violation, edge_coefficients, token_coefficients, rhs, coefficient_key = cut
        full_key = (*full_key_prefix, coefficient_key)
        if full_key in existing_cut_keys:
            continue

        entries = []
        entries.extend((col_idx, coefficient) for col_idx, coefficient in edge_coefficients.items())
        entries.extend(
            (t_offset + token_idx, coefficient)
            for token_idx, coefficient in token_coefficients.items()
        )
        violations.append((violation, full_key, entries, rhs))

    return violations


def separate_short_word_full_hull_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 250,
    max_word_length: int = 8,
    max_colors: int = 64,
    max_paths: int = 100000,
):
    """Separate full upward local-hull cuts for frequent short words.

    For selected local colours S, the integral local set contains one
    segmentation path plus any global token-activation superset of the colours
    used by that path. This separator searches arbitrary signed inequalities
    over edge-flow and t[S], with L1 coefficient normalization, that are valid
    for the whole upward local hull.
    """

    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g
    path_pattern_cache = {}

    for word_idx in rank_short_fractional_words(
        lp,
        f_values,
        t_values,
        max_words=max_words,
        max_word_length=max_word_length,
        tolerance=tolerance,
    ):
        selected_tokens = all_word_token_colors(lp, word_idx)
        if len(selected_tokens) < 2 or len(selected_tokens) > max_colors:
            continue
        full_key_prefix = ("short_word_full_hull", word_idx, selected_tokens)
        if any(key[:3] == full_key_prefix for key in existing_cut_keys):
            continue

        paths = enumerate_word_edge_paths_by_pattern(
            lp,
            word_idx,
            max_paths=max_paths,
            cache=path_pattern_cache,
        )
        if paths is None or len(paths) < 2:
            continue
        cut = full_upward_hull_cut_from_paths(
            lp,
            f_values,
            g_values,
            t_values,
            word_idx,
            selected_tokens,
            paths,
            tolerance=tolerance,
        )
        if cut is None:
            continue
        violation, edge_coefficients, token_coefficients, rhs, coefficient_key = cut
        full_key = (*full_key_prefix, coefficient_key)
        if full_key in existing_cut_keys:
            continue

        entries = []
        entries.extend((col_idx, coefficient) for col_idx, coefficient in edge_coefficients.items())
        entries.extend(
            (t_offset + token_idx, coefficient)
            for token_idx, coefficient in token_coefficients.items()
        )
        violations.append((violation, full_key, entries, rhs))

    return violations


def separate_short_word_pair_hull_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 500,
    max_word_length: int = 12,
    max_colors: int = 96,
    max_pair_rows: int = 250000,
    max_pairs: int = 800,
    top_words_per_color: int = 36,
    candidate_word_multiplier: float = 1.0,
    candidate_top_words_multiplier: float = 1.0,
    candidate_strategy: str = "score",
    candidate_random_seed: int = 0,
    pruning: str = "full",
    max_paths: int = 100000,
    workers: int = 0,
    batch_size: int = 32,
    min_fractional_shared_colors: int = 1,
    solution_cache: dict[str, dict | None] | None = None,
    solution_cache_max_entries: int = 500000,
    solution_cache_value_quantum: float = 1e-4,
):
    """Separate full upward local-hull cuts for pairs of short words."""

    start_time = time.monotonic()
    if pruning not in {"full", "fractional_edges_shared_colors"}:
        raise ValueError(f"Unsupported short_word_pair_hull pruning mode {pruning!r}")
    candidate_max_words = max(max_words, int(math.ceil(max_words * max(1.0, float(candidate_word_multiplier)))))
    candidate_top_words_per_color = max(
        top_words_per_color,
        int(math.ceil(top_words_per_color * max(1.0, float(candidate_top_words_multiplier)))),
    )
    pair_rows, word_color_scores = short_word_pair_candidates(
        lp,
        f_values,
        t_values,
        max_words=candidate_max_words,
        max_word_length=max_word_length,
        top_words_per_color=candidate_top_words_per_color,
        tolerance=tolerance,
    )
    pair_rows = reorder_short_word_pair_candidates(
        pair_rows,
        lp,
        t_values,
        max_pairs=max_pairs,
        tolerance=tolerance,
        strategy=candidate_strategy,
        random_seed=candidate_random_seed,
    )
    if not pair_rows:
        return []

    path_pattern_cache = {}
    paths_by_word = {}
    colors_by_word = {}
    tasks = []
    violations = []
    skipped_colors = 0
    skipped_paths = 0
    skipped_rows = 0
    skipped_shared = 0
    cache_hits = 0
    cache_cut_hits = 0
    cache_no_cut_hits = 0
    cache_full_skips = 0

    solution_cache = solution_cache if solution_cache is not None and solution_cache_max_entries > 0 else None
    existing_pair_prefixes = {
        key[:3]
        for key in existing_cut_keys
        if len(key) >= 3 and key[0] == "short_word_pair_hull"
    }

    for _, left_word, right_word in pair_rows:
        if len(tasks) >= max_pairs:
            break
        full_key_prefix = ("short_word_pair_hull", left_word, right_word)
        if full_key_prefix in existing_pair_prefixes:
            continue
        left_colors = colors_by_word.setdefault(left_word, all_word_token_colors(lp, left_word))
        right_colors = colors_by_word.setdefault(right_word, all_word_token_colors(lp, right_word))
        shared_fractional_colors = [
            token_idx
            for token_idx in set(left_colors) & set(right_colors)
            if tolerance < float(t_values[token_idx]) < 1.0 - tolerance
        ]
        if len(shared_fractional_colors) < min_fractional_shared_colors:
            skipped_shared += 1
            continue
        if pruning == "fractional_edges_shared_colors":
            selected_tokens = tuple(sorted(shared_fractional_colors))
        else:
            selected_tokens = tuple(sorted(set(left_colors) | set(right_colors)))
        if len(selected_tokens) < 2 or len(selected_tokens) > max_colors:
            skipped_colors += 1
            continue

        cache_key = None
        if solution_cache is not None:
            cache_key = pair_hull_projection_cache_key(
                lp,
                f_values,
                g_values,
                t_values,
                left_word,
                right_word,
                selected_tokens,
                pruning=pruning,
                value_quantum=solution_cache_value_quantum,
            )
            if cache_key in solution_cache:
                cache_hits += 1
                cached = solution_cache[cache_key]
                if cached is None or cached.get("cut") is None:
                    cache_no_cut_hits += 1
                else:
                    cut = cached["cut"]
                    current_violation = cut_violation_from_entries(
                        lp,
                        f_values,
                        g_values,
                        t_values,
                        cut[2],
                        cut[3],
                    )
                    if current_violation > tolerance and cut[1] not in existing_cut_keys:
                        cache_cut_hits += 1
                        violations.append((current_violation, cut[1], cut[2], cut[3]))
                continue
            if len(solution_cache) >= solution_cache_max_entries:
                cache_full_skips += 1

        if left_word not in paths_by_word:
            paths_by_word[left_word] = enumerate_word_edge_paths_by_pattern(
                lp,
                left_word,
                max_paths=max_paths,
                cache=path_pattern_cache,
            )
        if right_word not in paths_by_word:
            paths_by_word[right_word] = enumerate_word_edge_paths_by_pattern(
                lp,
                right_word,
                max_paths=max_paths,
                cache=path_pattern_cache,
            )
        left_paths = paths_by_word[left_word]
        right_paths = paths_by_word[right_word]
        if left_paths is None or right_paths is None:
            skipped_paths += 1
            continue
        if pruning == "full" and len(left_paths) * len(right_paths) > max_pair_rows:
            skipped_rows += 1
            continue
        tasks.append((left_word, right_word, selected_tokens, cache_key))

    if not tasks:
        LOGGER.info(
            "short_word_pair_hull: no tasks after filtering candidates=%d candidate_words=%d "
            "candidate_top_words=%d candidate_strategy=%s candidate_seed=%d pruning=%s skipped_colors=%d "
            "skipped_shared=%d skipped_paths=%d skipped_rows=%d cache_hits=%d cached_cuts=%d "
            "cached_no_cuts=%d cache_size=%d cache_full_skips=%d cache_value_quantum=%.6g",
            len(pair_rows),
            candidate_max_words,
            candidate_top_words_per_color,
            candidate_strategy,
            candidate_random_seed,
            pruning,
            skipped_colors,
            skipped_shared,
            skipped_paths,
            skipped_rows,
            cache_hits,
            cache_cut_hits,
            cache_no_cut_hits,
            len(solution_cache) if solution_cache is not None else 0,
            cache_full_skips,
            solution_cache_value_quantum,
        )
        return violations

    worker_count = resolve_pair_hull_workers(workers)
    worker_state = {
        "lp": lp,
        "f_values": f_values,
        "g_values": g_values,
        "t_values": t_values,
        "paths_by_word": paths_by_word,
        "tolerance": tolerance,
        "pruning": pruning,
        "max_pair_rows": max_pair_rows,
    }
    checked = 0
    build_seconds = 0.0
    solve_seconds = 0.0
    validation_seconds = 0.0
    worker_reduced_row_skips = 0
    reduced_rows = []
    edge_vars = []
    progress_interval = max(1, int(math.ceil(len(tasks) / 40)))
    next_progress = progress_interval

    def log_pair_hull_progress():
        nonlocal next_progress
        if checked < next_progress or checked >= len(tasks):
            return
        while next_progress <= checked:
            next_progress += progress_interval
        LOGGER.info(
            "short_word_pair_hull progress: checked=%d/%d %.1f%% cuts=%d workers=%d "
            "wall=%.3fs worker_build=%.3fs worker_solve=%.3fs worker_validate=%.3fs "
            "worker_row_skips=%d",
            checked,
            len(tasks),
            100.0 * checked / max(1, len(tasks)),
            len(violations),
            worker_count,
            time.monotonic() - start_time,
            build_seconds,
            solve_seconds,
            validation_seconds,
            worker_reduced_row_skips,
        )

    if worker_count == 1:
        init_pair_hull_worker(worker_state)
        for task in tasks:
            result = pair_hull_worker(task)
            if result is None:
                continue
            checked += 1
            build_seconds += result["build_seconds"]
            solve_seconds += result["solve_seconds"]
            validation_seconds += result.get("validation_seconds", 0.0)
            if result.get("skip_reason") == "reduced_rows":
                worker_reduced_row_skips += 1
            if "reduced_rows" in result:
                reduced_rows.append(result["reduced_rows"])
            if "edge_vars" in result:
                edge_vars.append(result["edge_vars"])
            maybe_cache_pair_hull_result(
                solution_cache,
                result.get("cache_key"),
                result if result["cut"] is not None else None,
                solution_cache_max_entries,
            )
            cut = result["cut"]
            if cut is not None:
                violations.append(cut)
            log_pair_hull_progress()
    else:
        context = mp.get_context("fork") if hasattr(os, "fork") else None
        task_batches = list(chunked(tasks, batch_size))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=init_pair_hull_worker,
            initargs=(worker_state,),
        ) as executor:
            futures = [executor.submit(pair_hull_batch_worker, batch) for batch in task_batches]
            for future in as_completed(futures):
                for result in future.result():
                    if result is None:
                        continue
                    checked += 1
                    build_seconds += result["build_seconds"]
                    solve_seconds += result["solve_seconds"]
                    validation_seconds += result.get("validation_seconds", 0.0)
                    if result.get("skip_reason") == "reduced_rows":
                        worker_reduced_row_skips += 1
                    if "reduced_rows" in result:
                        reduced_rows.append(result["reduced_rows"])
                    if "edge_vars" in result:
                        edge_vars.append(result["edge_vars"])
                    maybe_cache_pair_hull_result(
                        solution_cache,
                        result.get("cache_key"),
                        result if result["cut"] is not None else None,
                        solution_cache_max_entries,
                    )
                    cut = result["cut"]
                    if cut is not None:
                        violations.append(cut)
                log_pair_hull_progress()

    elapsed = time.monotonic() - start_time
    LOGGER.info(
        "short_word_pair_hull: candidates=%d candidate_words=%d candidate_top_words=%d "
        "candidate_strategy=%s candidate_seed=%d pruning=%s tasks=%d checked=%d cuts=%d workers=%d "
        "batches=%d batch_size=%d wall=%.3fs worker_build=%.3fs worker_solve=%.3fs "
        "worker_validate=%.3fs skipped_colors=%d skipped_shared=%d skipped_paths=%d skipped_rows=%d "
        "worker_row_skips=%d cache_hits=%d cached_cuts=%d cached_no_cuts=%d cache_size=%d "
        "cache_full_skips=%d cache_value_quantum=%.6g patterns=%d reduced_rows=%s edge_vars=%s",
        len(pair_rows),
        candidate_max_words,
        candidate_top_words_per_color,
        candidate_strategy,
        candidate_random_seed,
        pruning,
        len(tasks),
        checked,
        len(violations),
        worker_count,
        len(list(chunked(tasks, batch_size))) if worker_count > 1 else len(tasks),
        max(1, int(batch_size)),
        elapsed,
        build_seconds,
        solve_seconds,
        validation_seconds,
        skipped_colors,
        skipped_shared,
        skipped_paths,
        skipped_rows,
        worker_reduced_row_skips,
        cache_hits,
        cache_cut_hits,
        cache_no_cut_hits,
        len(solution_cache) if solution_cache is not None else 0,
        cache_full_skips,
        solution_cache_value_quantum,
        len(path_pattern_cache),
        small_quantile_summary(reduced_rows),
        small_quantile_summary(edge_vars),
    )
    return violations


def separate_short_word_triple_hull_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 700,
    max_word_length: int = 12,
    max_rows: int = 200000,
    max_triples: int = 30000,
    top_words_per_color: int = 48,
    candidate_word_multiplier: float = 1.0,
    candidate_top_words_multiplier: float = 1.0,
    candidate_sample: int = 250000,
    candidate_random_seed: int = 0,
    token_mode: str = "at_least_two",
    max_paths: int = 100000,
    workers: int = 0,
    batch_size: int = 32,
    min_fractional_colors: int = 2,
):
    """Separate reduced local-hull cuts for triples of short words."""

    start_time = time.monotonic()
    if token_mode not in {"shared_all", "at_least_two"}:
        raise ValueError(f"Unsupported short_word_triple_hull token mode {token_mode!r}")
    rng = random.Random(candidate_random_seed)
    candidate_max_words = max(max_words, int(math.ceil(max_words * max(1.0, float(candidate_word_multiplier)))))
    candidate_top_words_per_color = max(
        top_words_per_color,
        int(math.ceil(top_words_per_color * max(1.0, float(candidate_top_words_multiplier)))),
    )
    fractional_colors = set(np.flatnonzero((t_values > tolerance) & (t_values < 1.0 - tolerance)))
    ranked_words = rank_short_fractional_words(
        lp,
        f_values,
        t_values,
        max_words=candidate_max_words,
        max_word_length=max_word_length,
        tolerance=tolerance,
    )
    word_colors = {}
    color_to_words = defaultdict(list)
    for word_idx in ranked_words:
        scores = defaultdict(float)
        word_weight = float(lp["word_weights"][word_idx])
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            edge_value = float(f_values[edge_idx])
            if not (tolerance < edge_value < 1.0 - tolerance):
                continue
            info = lp["nonfree_edge_info"][edge_idx]
            token_idx = info["token_index"]
            if token_idx not in fractional_colors:
                continue
            scores[token_idx] += min(edge_value, 1.0 - edge_value) * max(1, info["end"] - info["start"])
        word_colors[word_idx] = set(all_word_token_colors(lp, word_idx))
        for token_idx, score in scores.items():
            color_to_words[token_idx].append((word_weight * score, word_idx))
    for rows in color_to_words.values():
        rows.sort(reverse=True)

    triple_candidates = set()
    estimated_candidate_triples = 0
    colors = list(color_to_words)
    rng.shuffle(colors)
    per_color_budget = max(1, int(math.ceil(max(1, candidate_sample) / max(1, len(colors)))))
    for token_idx in colors:
        words = [word_idx for _, word_idx in color_to_words[token_idx][:candidate_top_words_per_color]]
        if len(words) < 3:
            continue
        estimated_candidate_triples += math.comb(len(words), 3)
        if len(triple_candidates) >= candidate_sample:
            continue
        for _ in range(per_color_budget):
            triple_candidates.add(tuple(sorted(rng.sample(words, 3))))
            if len(triple_candidates) >= candidate_sample:
                break

    candidates = list(triple_candidates)
    rng.shuffle(candidates)
    existing_triple_prefixes = {
        key[:4]
        for key in existing_cut_keys
        if len(key) >= 4 and key[0] == "short_word_triple_hull"
    }
    path_pattern_cache = {}
    paths_by_word = {}
    tasks = []
    skipped = Counter()
    for words3 in candidates:
        if len(tasks) >= max_triples:
            break
        full_key_prefix = ("short_word_triple_hull", *words3)
        if full_key_prefix in existing_triple_prefixes:
            skipped["existing_cut"] += 1
            continue
        color_sets = [word_colors.setdefault(word_idx, set(all_word_token_colors(lp, word_idx))) for word_idx in words3]
        if token_mode == "shared_all":
            selected_set = set.intersection(*color_sets) & fractional_colors
        else:
            counts = Counter(token_idx for colors_for_word in color_sets for token_idx in colors_for_word if token_idx in fractional_colors)
            selected_set = {token_idx for token_idx, count in counts.items() if count >= 2}
        selected_tokens = tuple(sorted(selected_set))
        if len(selected_tokens) < min_fractional_colors:
            skipped["shared"] += 1
            continue
        ok = True
        for word_idx in words3:
            if word_idx not in paths_by_word:
                paths_by_word[word_idx] = enumerate_word_edge_paths_by_pattern(
                    lp,
                    word_idx,
                    max_paths=max_paths,
                    cache=path_pattern_cache,
                )
            if paths_by_word[word_idx] is None:
                ok = False
                break
        if not ok:
            skipped["paths"] += 1
            continue
        tasks.append((*words3, selected_tokens))

    if not tasks:
        LOGGER.info(
            "short_word_triple_hull: no tasks after filtering candidates_sampled=%d "
            "estimated_candidates=%d candidate_words=%d candidate_top_words=%d candidate_seed=%d "
            "token_mode=%s skipped=%s patterns=%d",
            len(candidates),
            estimated_candidate_triples,
            candidate_max_words,
            candidate_top_words_per_color,
            candidate_random_seed,
            token_mode,
            dict(skipped),
            len(path_pattern_cache),
        )
        return []

    worker_count = resolve_pair_hull_workers(workers)
    worker_state = {
        "lp": lp,
        "f_values": f_values,
        "g_values": g_values,
        "t_values": t_values,
        "paths_by_word": paths_by_word,
        "tolerance": tolerance,
        "max_rows": max_rows,
    }
    violations = []
    checked = 0
    build_seconds = 0.0
    solve_seconds = 0.0
    validation_seconds = 0.0
    worker_reduced_row_skips = 0
    reduced_rows = []
    edge_vars = []
    progress_interval = max(1, int(math.ceil(len(tasks) / 40)))
    next_progress = progress_interval

    def handle_result(result):
        nonlocal checked, build_seconds, solve_seconds, validation_seconds, worker_reduced_row_skips
        if result is None:
            return
        checked += 1
        build_seconds += result["build_seconds"]
        solve_seconds += result["solve_seconds"]
        validation_seconds += result.get("validation_seconds", 0.0)
        if result.get("skip_reason") == "reduced_rows":
            worker_reduced_row_skips += 1
        if "reduced_rows" in result:
            reduced_rows.append(result["reduced_rows"])
        if "edge_vars" in result:
            edge_vars.append(result["edge_vars"])
        cut = result["cut"]
        if cut is not None:
            violations.append(cut)

    def log_triple_hull_progress():
        nonlocal next_progress
        if checked < next_progress or checked >= len(tasks):
            return
        while next_progress <= checked:
            next_progress += progress_interval
        LOGGER.info(
            "short_word_triple_hull progress: checked=%d/%d %.1f%% cuts=%d workers=%d "
            "wall=%.3fs worker_build=%.3fs worker_solve=%.3fs worker_validate=%.3fs "
            "worker_row_skips=%d",
            checked,
            len(tasks),
            100.0 * checked / max(1, len(tasks)),
            len(violations),
            worker_count,
            time.monotonic() - start_time,
            build_seconds,
            solve_seconds,
            validation_seconds,
            worker_reduced_row_skips,
        )

    if worker_count == 1:
        init_triple_hull_worker(worker_state)
        for task in tasks:
            handle_result(triple_hull_worker(task))
            log_triple_hull_progress()
    else:
        context = mp.get_context("fork") if hasattr(os, "fork") else None
        task_batches = list(chunked(tasks, batch_size))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=init_triple_hull_worker,
            initargs=(worker_state,),
        ) as executor:
            futures = [executor.submit(triple_hull_batch_worker, batch) for batch in task_batches]
            for future in as_completed(futures):
                for result in future.result():
                    handle_result(result)
                log_triple_hull_progress()

    elapsed = time.monotonic() - start_time
    LOGGER.info(
        "short_word_triple_hull: candidates_sampled=%d estimated_candidates=%d candidate_words=%d "
        "candidate_top_words=%d candidate_seed=%d token_mode=%s tasks=%d checked=%d cuts=%d "
        "workers=%d batches=%d batch_size=%d wall=%.3fs worker_build=%.3fs worker_solve=%.3fs "
        "worker_validate=%.3fs skipped=%s worker_row_skips=%d patterns=%d reduced_rows=%s edge_vars=%s",
        len(candidates),
        estimated_candidate_triples,
        candidate_max_words,
        candidate_top_words_per_color,
        candidate_random_seed,
        token_mode,
        len(tasks),
        checked,
        len(violations),
        worker_count,
        len(list(chunked(tasks, batch_size))) if worker_count > 1 else len(tasks),
        max(1, int(batch_size)),
        elapsed,
        build_seconds,
        solve_seconds,
        validation_seconds,
        dict(skipped),
        worker_reduced_row_skips,
        len(path_pattern_cache),
        small_quantile_summary(reduced_rows),
        small_quantile_summary(edge_vars),
    )
    return violations


def separate_short_word_pair_chain_template_cut_specs(
    lp,
    f_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    family_name: str,
    template: str,
    max_words: int = 700,
    max_word_length: int = 12,
    candidate_word_multiplier: float = 1.0,
    top_supports_per_shape: int = 24,
    max_chain_edges: int = 6,
    max_template_cuts: int = 100000,
    max_paths: int = 100000,
):
    """Separate validated pair-chain templates found by brute-force pair LPs."""

    start_time = time.monotonic()
    if template not in {"single", "bridge", "both"}:
        raise ValueError(f"Unsupported short_word_pair_chain template {template!r}")
    candidate_max_words = max(max_words, int(math.ceil(max_words * max(1.0, float(candidate_word_multiplier)))))
    ranked_words = rank_short_fractional_words(
        lp,
        f_values,
        t_values,
        max_words=candidate_max_words,
        max_word_length=max_word_length,
        tolerance=tolerance,
    )
    single_supports, bridge_supports, skipped = collect_short_word_pair_chain_supports(
        lp,
        f_values,
        t_values,
        ranked_words,
        tolerance=tolerance,
        top_supports_per_shape=top_supports_per_shape,
        max_chain_edges=max_chain_edges,
    )
    violations = []
    candidates = 0
    path_skips = 0
    if template in {"single", "both"}:
        single_violations, single_candidates, single_path_skips = separate_single_chain_template_cuts(
            lp,
            f_values,
            t_values,
            single_supports,
            existing_cut_keys=existing_cut_keys,
            family_name=family_name,
            tolerance=tolerance,
            max_template_cuts=max_template_cuts,
            max_paths=max_paths,
        )
        violations.extend(single_violations)
        candidates += single_candidates
        path_skips += single_path_skips
    if template in {"bridge", "both"}:
        bridge_violations, bridge_candidates, bridge_path_skips = separate_bridge_chain_template_cuts(
            lp,
            f_values,
            t_values,
            bridge_supports,
            existing_cut_keys=existing_cut_keys,
            family_name=family_name,
            tolerance=tolerance,
            max_template_cuts=max_template_cuts,
            max_paths=max_paths,
        )
        violations.extend(bridge_violations)
        candidates += bridge_candidates
        path_skips += bridge_path_skips
    if max_template_cuts > 0 and len(violations) > max_template_cuts:
        violations.sort(key=lambda item: item[0], reverse=True)
        violations = violations[:max_template_cuts]
    LOGGER.info(
        "%s: template=%s candidate_words=%d single_supports=%d bridge_shapes=%d "
        "candidates=%d cuts=%d path_skips=%d skipped=%s top_supports_per_shape=%d "
        "max_chain_edges=%d max_template_cuts=%d wall=%.3fs",
        family_name,
        template,
        candidate_max_words,
        len(single_supports),
        len(bridge_supports),
        candidates,
        len(violations),
        path_skips,
        dict(skipped),
        top_supports_per_shape,
        max_chain_edges,
        max_template_cuts,
        time.monotonic() - start_time,
    )
    return violations


def collect_short_word_pair_chain_supports(
    lp,
    f_values,
    t_values,
    word_indices,
    *,
    tolerance: float,
    top_supports_per_shape: int,
    max_chain_edges: int,
):
    single_supports = []
    bridge_by_shape_word = {}
    skipped = Counter()
    for word_idx in word_indices:
        edges_by_start = defaultdict(list)
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            edge_value = float(f_values[edge_idx])
            if not (tolerance < edge_value < 1.0 - tolerance):
                continue
            info = lp["nonfree_edge_info"][edge_idx]
            token_idx = int(info["token_index"])
            token_value = float(t_values[token_idx])
            if not (tolerance < token_value < 1.0 - tolerance):
                continue
            edges_by_start[int(info["start"])].append(
                {
                    "col": int(edge_idx),
                    "token": token_idx,
                    "start": int(info["start"]),
                    "end": int(info["end"]),
                    "value": edge_value,
                }
            )
        if not edges_by_start:
            skipped["no_edges"] += 1
            continue
        for start in list(edges_by_start):
            edges_by_start[start].sort(key=lambda edge: edge["value"], reverse=True)
            del edges_by_start[start][4:]

        for start in sorted(edges_by_start):
            for edge in edges_by_start[start]:
                visit_pair_chain_supports(
                    word_idx,
                    (edge,),
                    edges_by_start,
                    single_supports,
                    bridge_by_shape_word,
                    max_chain_edges=max_chain_edges,
                )

    bridge_supports = finalize_template_supports(bridge_by_shape_word, top_supports_per_shape)
    single_supports.sort(key=lambda row: row["value"], reverse=True)
    return single_supports[: max(1, int(top_supports_per_shape)) * max(1, len(word_indices))], bridge_supports, skipped


def visit_pair_chain_supports(
    word_idx,
    chain,
    edges_by_start,
    single_supports,
    bridge_by_shape_word,
    *,
    max_chain_edges,
):
    if len(chain) >= 2:
        first_token = int(chain[0]["token"])
        last_token = int(chain[-1]["token"])
        columns = tuple(int(edge["col"]) for edge in chain)
        support = {
            "word": int(word_idx),
            "columns": columns,
            "tokens": tuple(sorted({first_token, last_token})),
            "value": float(sum(float(edge["value"]) for edge in chain)),
            "shape": tuple(
                (
                    int(edge["start"] - chain[0]["start"]),
                    int(edge["end"] - chain[0]["start"]),
                )
                for edge in chain
            ),
        }
        if first_token == last_token and len(chain) >= 4:
            single_supports.append(support)
        elif first_token != last_token:
            keep_best_support_for_word(bridge_by_shape_word, tuple(sorted((first_token, last_token))), word_idx, support)
    if len(chain) >= max_chain_edges:
        return
    next_start = int(chain[-1]["start"]) + 1
    for edge in edges_by_start.get(next_start, ()):
        if intervals_overlap(chain[-1], edge):
            visit_pair_chain_supports(
                word_idx,
                (*chain, edge),
                edges_by_start,
                single_supports,
                bridge_by_shape_word,
                max_chain_edges=max_chain_edges,
            )


def separate_single_chain_template_cuts(
    lp,
    f_values,
    t_values,
    single_supports,
    *,
    existing_cut_keys,
    family_name,
    tolerance,
    max_template_cuts,
    max_paths,
):
    t_offset = lp["num_nonfree_edges"] + lp["num_free_edges"]
    heap = []
    sequence = 0
    candidates = 0
    path_skips = 0
    emitted = set()
    path_pattern_cache = {}
    paths_by_word = {}
    for support in single_supports:
        candidates += 1
        word_idx = int(support["word"])
        selected_tokens = tuple(int(token_idx) for token_idx in support["tokens"])
        columns = tuple(int(col_idx) for col_idx in support["columns"])
        if len(selected_tokens) != 1:
            continue
        coefficient_key = (word_idx, selected_tokens, columns)
        if coefficient_key in emitted:
            continue
        full_key = (family_name, "single", word_idx, selected_tokens, columns)
        if full_key in existing_cut_keys:
            continue
        edge_coefficients = {col_idx: 1.0 for col_idx in columns}
        token_coefficients = {selected_tokens[0]: -1.0}
        rhs = template_cut_max_lhs(
            lp,
            (word_idx,),
            selected_tokens,
            edge_coefficients,
            token_coefficients,
            max_paths=max_paths,
            path_pattern_cache=path_pattern_cache,
            paths_by_word=paths_by_word,
        )
        if rhs is None:
            path_skips += 1
            continue
        violation = float(sum(float(f_values[col_idx]) for col_idx in columns) - float(t_values[selected_tokens[0]]) - rhs)
        if violation <= tolerance:
            continue
        emitted.add(coefficient_key)
        entries = [(col_idx, 1.0) for col_idx in columns]
        entries.append((t_offset + selected_tokens[0], -1.0))
        cut = (float(violation), full_key, entries, float(rhs))
        if max_template_cuts > 0:
            if len(heap) < max_template_cuts:
                heapq.heappush(heap, (float(violation), sequence, cut))
                sequence += 1
            elif violation > heap[0][0]:
                heapq.heapreplace(heap, (float(violation), sequence, cut))
                sequence += 1
        else:
            heap.append((float(violation), sequence, cut))
            sequence += 1
    violations = [row[2] for row in sorted(heap, key=lambda item: item[0], reverse=True)]
    return violations, candidates, path_skips


def separate_bridge_chain_template_cuts(
    lp,
    f_values,
    t_values,
    bridge_supports,
    *,
    existing_cut_keys,
    family_name,
    tolerance,
    max_template_cuts,
    max_paths,
):
    t_offset = lp["num_nonfree_edges"] + lp["num_free_edges"]
    heap = []
    sequence = 0
    candidates = 0
    path_skips = 0
    emitted = set()
    path_pattern_cache = {}
    paths_by_word = {}
    for selected_tokens, supports in bridge_supports.items():
        if len(selected_tokens) != 2:
            continue
        token_sum = sum(float(t_values[token_idx]) for token_idx in selected_tokens)
        for left_idx, left_support in enumerate(supports):
            for right_support in supports[left_idx + 1 :]:
                left_word = int(left_support["word"])
                right_word = int(right_support["word"])
                if left_word == right_word:
                    continue
                candidates += 1
                words = (left_word, right_word)
                columns = tuple(int(col_idx) for col_idx in (*left_support["columns"], *right_support["columns"]))
                coefficient_key = (tuple(sorted(words)), tuple(int(token_idx) for token_idx in selected_tokens), tuple(sorted(columns)))
                if coefficient_key in emitted:
                    continue
                full_key = (family_name, "bridge", tuple(sorted(words)), tuple(int(token_idx) for token_idx in selected_tokens), tuple(sorted(columns)))
                if full_key in existing_cut_keys:
                    continue
                edge_coefficients = {col_idx: 1.0 for col_idx in columns}
                token_coefficients = {int(token_idx): -1.0 for token_idx in selected_tokens}
                rhs = template_cut_max_lhs(
                    lp,
                    words,
                    selected_tokens,
                    edge_coefficients,
                    token_coefficients,
                    max_paths=max_paths,
                    path_pattern_cache=path_pattern_cache,
                    paths_by_word=paths_by_word,
                )
                if rhs is None:
                    path_skips += 1
                    continue
                violation = float(sum(float(f_values[col_idx]) for col_idx in columns) - token_sum - rhs)
                if violation <= tolerance:
                    continue
                emitted.add(coefficient_key)
                entries = [(col_idx, 1.0) for col_idx in columns]
                entries.extend((t_offset + token_idx, -1.0) for token_idx in selected_tokens)
                cut = (float(violation), full_key, entries, float(rhs))
                if max_template_cuts > 0:
                    if len(heap) < max_template_cuts:
                        heapq.heappush(heap, (float(violation), sequence, cut))
                        sequence += 1
                    elif violation > heap[0][0]:
                        heapq.heapreplace(heap, (float(violation), sequence, cut))
                        sequence += 1
                else:
                    heap.append((float(violation), sequence, cut))
                    sequence += 1
    violations = [row[2] for row in sorted(heap, key=lambda item: item[0], reverse=True)]
    return violations, candidates, path_skips


def separate_short_word_triple_template_cut_specs(
    lp,
    f_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    family_name: str,
    template: str,
    max_words: int = 700,
    max_word_length: int = 12,
    candidate_word_multiplier: float = 1.0,
    top_supports_per_shape: int = 24,
    max_template_cuts: int = 100000,
    validate_templates: bool = False,
):
    """Separate fixed small hypergraph templates found by the triplet LP separator."""

    start_time = time.monotonic()
    if template not in {"triangle", "4cycle"}:
        raise ValueError(f"Unsupported short_word_triple template {template!r}")
    candidate_max_words = max(max_words, int(math.ceil(max_words * max(1.0, float(candidate_word_multiplier)))))
    ranked_words = rank_short_fractional_words(
        lp,
        f_values,
        t_values,
        max_words=candidate_max_words,
        max_word_length=max_word_length,
        tolerance=tolerance,
    )
    pair_supports, triple_supports, skipped = collect_short_word_template_supports(
        lp,
        f_values,
        t_values,
        ranked_words,
        tolerance=tolerance,
        top_supports_per_shape=top_supports_per_shape,
        need_triples=template == "4cycle",
    )
    if template == "triangle":
        violations, candidates, invalid = separate_triangle_template_cuts(
            lp,
            f_values,
            t_values,
            pair_supports,
            max_paths=100000,
            existing_cut_keys=existing_cut_keys,
            family_name=family_name,
            tolerance=tolerance,
            max_template_cuts=max_template_cuts,
            validate_templates=validate_templates,
        )
    else:
        violations, candidates, invalid = separate_4cycle_template_cuts(
            lp,
            f_values,
            t_values,
            pair_supports,
            triple_supports,
            max_paths=100000,
            existing_cut_keys=existing_cut_keys,
            family_name=family_name,
            tolerance=tolerance,
            max_template_cuts=max_template_cuts,
            validate_templates=validate_templates,
        )
    LOGGER.info(
        "%s: template=%s candidate_words=%d pair_shapes=%d triple_shapes=%d candidates=%d "
        "cuts=%d invalid=%d skipped=%s top_supports_per_shape=%d max_template_cuts=%d "
        "validate=%s wall=%.3fs",
        family_name,
        template,
        candidate_max_words,
        len(pair_supports),
        len(triple_supports),
        candidates,
        len(violations),
        invalid,
        dict(skipped),
        top_supports_per_shape,
        max_template_cuts,
        validate_templates,
        time.monotonic() - start_time,
    )
    return violations


def collect_short_word_template_supports(
    lp,
    f_values,
    t_values,
    word_indices,
    *,
    tolerance: float,
    top_supports_per_shape: int,
    need_triples: bool,
):
    pair_by_shape_word = {}
    triple_by_shape_word = {}
    skipped = Counter()
    for word_idx in word_indices:
        edges = []
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            edge_value = float(f_values[edge_idx])
            if not (tolerance < edge_value < 1.0 - tolerance):
                continue
            info = lp["nonfree_edge_info"][edge_idx]
            token_idx = int(info["token_index"])
            token_value = float(t_values[token_idx])
            if not (tolerance < token_value < 1.0 - tolerance):
                continue
            edges.append(
                {
                    "col": int(edge_idx),
                    "token": token_idx,
                    "start": int(info["start"]),
                    "end": int(info["end"]),
                    "value": edge_value,
                }
            )
        if len(edges) < 2:
            skipped["few_edges"] += 1
            continue
        for left_idx, left in enumerate(edges):
            for right in edges[left_idx + 1 :]:
                if left["token"] == right["token"] or not intervals_overlap(left, right):
                    continue
                token_shape = tuple(sorted((left["token"], right["token"])))
                support = template_support(word_idx, (left, right))
                keep_best_support_for_word(pair_by_shape_word, token_shape, word_idx, support)
        if not need_triples or len(edges) < 3:
            continue
        for first_idx, first in enumerate(edges):
            for second_idx in range(first_idx + 1, len(edges)):
                second = edges[second_idx]
                if first["token"] == second["token"] or not intervals_overlap(first, second):
                    continue
                for third in edges[second_idx + 1 :]:
                    if len({first["token"], second["token"], third["token"]}) < 3:
                        continue
                    if not (
                        intervals_overlap(first, third)
                        and intervals_overlap(second, third)
                    ):
                        continue
                    token_shape = tuple(sorted((first["token"], second["token"], third["token"])))
                    support = template_support(word_idx, (first, second, third))
                    keep_best_support_for_word(triple_by_shape_word, token_shape, word_idx, support)

    pair_supports = finalize_template_supports(pair_by_shape_word, top_supports_per_shape)
    triple_supports = finalize_template_supports(triple_by_shape_word, top_supports_per_shape)
    return pair_supports, triple_supports, skipped


def intervals_overlap(left, right) -> bool:
    return max(left["start"], right["start"]) < min(left["end"], right["end"])


def template_support(word_idx: int, edges):
    columns = tuple(sorted(int(edge["col"]) for edge in edges))
    return {
        "word": int(word_idx),
        "columns": columns,
        "value": float(sum(float(edge["value"]) for edge in edges)),
    }


def keep_best_support_for_word(by_shape_word, token_shape, word_idx, support):
    key = (token_shape, int(word_idx))
    previous = by_shape_word.get(key)
    if previous is None or support["value"] > previous["value"]:
        by_shape_word[key] = support


def finalize_template_supports(by_shape_word, top_supports_per_shape: int):
    by_shape = defaultdict(list)
    for token_shape, _word_idx in by_shape_word:
        by_shape[token_shape].append(by_shape_word[(token_shape, _word_idx)])
    limit = max(1, int(top_supports_per_shape))
    return {
        token_shape: sorted(supports, key=lambda row: row["value"], reverse=True)[:limit]
        for token_shape, supports in by_shape.items()
    }


def separate_triangle_template_cuts(
    lp,
    f_values,
    t_values,
    pair_supports,
    *,
    max_paths,
    existing_cut_keys,
    family_name,
    tolerance,
    max_template_cuts,
    validate_templates,
):
    t_offset = lp["num_nonfree_edges"] + lp["num_free_edges"]
    neighbors = defaultdict(set)
    for left, right in pair_supports:
        neighbors[left].add(right)
        neighbors[right].add(left)
    heap = []
    sequence = 0
    candidates = 0
    invalid = 0
    emitted = set()
    path_pattern_cache = {}
    paths_by_word = {}
    for first in sorted(neighbors):
        later = sorted(token_idx for token_idx in neighbors[first] if token_idx > first)
        for second_idx, second in enumerate(later):
            for third in later[second_idx + 1 :]:
                if (second, third) not in pair_supports:
                    continue
                token_shape = (first, second, third)
                token_sum = sum(float(t_values[token_idx]) for token_idx in token_shape)
                for first_support in pair_supports[(first, second)]:
                    for second_support in pair_supports[(first, third)]:
                        if second_support["word"] == first_support["word"]:
                            continue
                        for third_support in pair_supports[(second, third)]:
                            words = (first_support["word"], second_support["word"], third_support["word"])
                            if len(set(words)) < 3:
                                continue
                            candidates += 1
                            columns = first_support["columns"] + second_support["columns"] + third_support["columns"]
                            edge_sum = float(f_values[list(columns)].sum())
                            violation = edge_sum - token_sum - 1.0
                            if violation <= tolerance:
                                continue
                            if max_template_cuts > 0 and len(heap) >= max_template_cuts and violation <= heap[0][0]:
                                continue
                            coefficient_key = (token_shape, tuple(sorted(columns)))
                            if coefficient_key in emitted:
                                continue
                            full_key = (family_name, token_shape, tuple(sorted(words)), coefficient_key)
                            if full_key in existing_cut_keys:
                                continue
                            if validate_templates:
                                max_slack = validate_template_cut(
                                    lp,
                                    words,
                                    token_shape,
                                    columns,
                                    max_paths=max_paths,
                                    path_pattern_cache=path_pattern_cache,
                                    paths_by_word=paths_by_word,
                                )
                                if max_slack is None or max_slack > 1e-7:
                                    invalid += 1
                                    continue
                            emitted.add(coefficient_key)
                            entries = [(col_idx, 1.0) for col_idx in columns]
                            entries.extend((t_offset + token_idx, -1.0) for token_idx in token_shape)
                            cut = (float(violation), full_key, entries, 1.0)
                            if max_template_cuts > 0:
                                if len(heap) < max_template_cuts:
                                    heapq.heappush(heap, (float(violation), sequence, cut))
                                    sequence += 1
                                else:
                                    heapq.heapreplace(heap, (float(violation), sequence, cut))
                                    sequence += 1
                            else:
                                heap.append((float(violation), sequence, cut))
                                sequence += 1
    violations = [row[2] for row in sorted(heap, key=lambda item: item[0], reverse=True)]
    return violations, candidates, invalid


def separate_4cycle_template_cuts(
    lp,
    f_values,
    t_values,
    pair_supports,
    triple_supports,
    *,
    max_paths,
    existing_cut_keys,
    family_name,
    tolerance,
    max_template_cuts,
    validate_templates,
):
    t_offset = lp["num_nonfree_edges"] + lp["num_free_edges"]
    heap = []
    sequence = 0
    candidates = 0
    invalid = 0
    emitted = set()
    path_pattern_cache = {}
    paths_by_word = {}
    triple_shapes = sorted(triple_supports)
    for left_idx, left_shape in enumerate(triple_shapes):
        left_set = set(left_shape)
        for right_shape in triple_shapes[left_idx + 1 :]:
            right_set = set(right_shape)
            intersection = left_set & right_set
            if len(intersection) != 2:
                continue
            union_shape = tuple(sorted(left_set | right_set))
            if len(union_shape) != 4:
                continue
            pair_shape = tuple(sorted((left_set - intersection) | (right_set - intersection)))
            if pair_shape not in pair_supports:
                continue
            token_sum = sum(float(t_values[token_idx]) for token_idx in union_shape)
            for left_support in triple_supports[left_shape]:
                for right_support in triple_supports[right_shape]:
                    if right_support["word"] == left_support["word"]:
                        continue
                    for pair_support in pair_supports[pair_shape]:
                        words = (left_support["word"], right_support["word"], pair_support["word"])
                        if len(set(words)) < 3:
                            continue
                        candidates += 1
                        columns = left_support["columns"] + right_support["columns"] + pair_support["columns"]
                        edge_sum = float(f_values[list(columns)].sum())
                        violation = edge_sum - token_sum - 1.0
                        if violation <= tolerance:
                            continue
                        if max_template_cuts > 0 and len(heap) >= max_template_cuts and violation <= heap[0][0]:
                            continue
                        coefficient_key = (left_shape, right_shape, pair_shape, tuple(sorted(columns)))
                        if coefficient_key in emitted:
                            continue
                        full_key = (family_name, union_shape, tuple(sorted(words)), coefficient_key)
                        if full_key in existing_cut_keys:
                            continue
                        if validate_templates:
                            max_slack = validate_template_cut(
                                lp,
                                words,
                                union_shape,
                                columns,
                                max_paths=max_paths,
                                path_pattern_cache=path_pattern_cache,
                                paths_by_word=paths_by_word,
                            )
                            if max_slack is None or max_slack > 1e-7:
                                invalid += 1
                                continue
                        emitted.add(coefficient_key)
                        entries = [(col_idx, 1.0) for col_idx in columns]
                        entries.extend((t_offset + token_idx, -1.0) for token_idx in union_shape)
                        cut = (float(violation), full_key, entries, 1.0)
                        if max_template_cuts > 0:
                            if len(heap) < max_template_cuts:
                                heapq.heappush(heap, (float(violation), sequence, cut))
                                sequence += 1
                            else:
                                heapq.heapreplace(heap, (float(violation), sequence, cut))
                                sequence += 1
                        else:
                            heap.append((float(violation), sequence, cut))
                            sequence += 1
    violations = [row[2] for row in sorted(heap, key=lambda item: item[0], reverse=True)]
    return violations, candidates, invalid


def validate_template_cut(
    lp,
    word_indices,
    selected_tokens,
    columns,
    *,
    max_paths,
    path_pattern_cache,
    paths_by_word,
):
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    selected_set = set(selected_tokens)
    col_position = {int(col_idx): idx for idx, col_idx in enumerate(columns)}
    signatures = []
    for word_idx in word_indices:
        if word_idx not in paths_by_word:
            paths_by_word[word_idx] = enumerate_word_edge_paths_by_pattern(
                lp,
                word_idx,
                max_paths=max_paths,
                cache=path_pattern_cache,
            )
        paths = paths_by_word[word_idx]
        if paths is None:
            return None
        signatures.append(
            projected_pair_hull_path_signatures(
                paths,
                selected_set,
                selected_position,
                col_position,
            )
        )
    max_slack = -float("inf")
    for first_edges, first_mask in signatures[0]:
        for second_edges, second_mask in signatures[1]:
            pair_edges = first_edges + second_edges
            pair_mask = first_mask | second_mask
            for third_edges, third_mask in signatures[2]:
                token_mask = pair_mask | third_mask
                lhs = len(pair_edges) + len(third_edges) - int(token_mask.bit_count())
                max_slack = max(max_slack, float(lhs - 1.0))
                if max_slack > 1e-7:
                    return max_slack
    return max_slack


def template_cut_max_lhs(
    lp,
    word_indices,
    selected_tokens,
    edge_coefficients,
    token_coefficients,
    *,
    max_paths,
    path_pattern_cache,
    paths_by_word,
):
    selected_tokens = tuple(int(token_idx) for token_idx in selected_tokens)
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    selected_set = set(selected_tokens)
    columns = tuple(sorted(int(col_idx) for col_idx in edge_coefficients))
    col_position = {col_idx: idx for idx, col_idx in enumerate(columns)}
    signatures = []
    for word_idx in word_indices:
        word_idx = int(word_idx)
        if word_idx not in paths_by_word:
            paths_by_word[word_idx] = enumerate_word_edge_paths_by_pattern(
                lp,
                word_idx,
                max_paths=max_paths,
                cache=path_pattern_cache,
            )
        paths = paths_by_word[word_idx]
        if paths is None:
            return None
        signatures.append(
            projected_pair_hull_path_signatures(
                paths,
                selected_set,
                selected_position,
                col_position,
            )
        )

    positive_token_sum = sum(max(0.0, float(token_coefficients.get(token_idx, 0.0))) for token_idx in selected_tokens)
    max_lhs = -float("inf")

    def visit(signature_idx, kept_positions, token_mask):
        nonlocal max_lhs
        if signature_idx == len(signatures):
            lhs = positive_token_sum
            for pos in kept_positions:
                lhs += float(edge_coefficients.get(columns[pos], 0.0))
            for pos, token_idx in enumerate(selected_tokens):
                if token_mask & (1 << pos):
                    lhs += min(0.0, float(token_coefficients.get(token_idx, 0.0)))
            max_lhs = max(max_lhs, lhs)
            return
        for path_positions, path_mask in signatures[signature_idx]:
            visit(signature_idx + 1, kept_positions + path_positions, token_mask | path_mask)

    visit(0, tuple(), 0)
    return max_lhs


def word_support_selected_tokens(lp, f_values, t_values, word_idx: int, *, max_rank: int, tolerance: float):
    scores = defaultdict(float)
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        value = float(f_values[edge_idx])
        if value <= tolerance:
            continue
        info = lp["nonfree_edge_info"][edge_idx]
        token_idx = info["token_index"]
        token_value = float(t_values[token_idx])
        if tolerance < token_value < 1.0 - tolerance:
            scores[token_idx] += value * max(1, info["end"] - info["start"])
    selected = [
        token_idx
        for token_idx, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max_rank]
    ]
    return tuple(sorted(selected))


def enumerate_word_edge_paths(lp, word_idx: int, *, max_paths: int):
    num_f = lp["num_nonfree_edges"]
    by_start = defaultdict(list)
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        info = lp["nonfree_edge_info"][edge_idx]
        by_start[info["start"]].append((info["end"], edge_idx, info["token_index"]))
    for edge_idx in lp["word_free_edges"].get(word_idx, []):
        info = lp["free_edge_info"][edge_idx]
        by_start[info["start"]].append((info["end"], num_f + edge_idx, None))

    target = lp["word_lengths"][word_idx]
    paths = []

    def visit(position: int, columns: tuple[int, ...], token_set: frozenset[int]):
        if len(paths) > max_paths:
            return
        if position == target:
            paths.append((columns, token_set))
            return
        for end, col_idx, token_idx in by_start.get(position, []):
            next_set = token_set if token_idx is None else token_set | frozenset((token_idx,))
            visit(end, (*columns, col_idx), next_set)

    visit(0, tuple(), frozenset())
    if len(paths) > max_paths:
        return None
    return paths


def enumerate_word_edge_paths_by_pattern(lp, word_idx: int, *, max_paths: int, cache: dict):
    local_edges, pattern_key, token_class_to_index = word_edge_pattern(lp, word_idx)
    cached_paths = cache.get(pattern_key)
    if cached_paths is None and pattern_key not in cache:
        cached_paths = enumerate_local_pattern_paths(
            local_edges,
            lp["word_lengths"][word_idx],
            max_paths=max_paths,
        )
        cache[pattern_key] = cached_paths
    if cached_paths is None:
        return None

    paths = []
    for local_indices, token_classes in cached_paths:
        columns = tuple(local_edges[local_idx]["col"] for local_idx in local_indices)
        token_set = frozenset(token_class_to_index[token_class] for token_class in token_classes)
        paths.append((columns, token_set))
    return paths


def word_edge_pattern(lp, word_idx: int):
    num_f = lp["num_nonfree_edges"]
    token_class_by_string = {}
    local_edges = []

    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        info = lp["nonfree_edge_info"][edge_idx]
        token = info["token"]
        token_class = token_class_by_string.setdefault(token, len(token_class_by_string))
        local_edges.append(
            {
                "start": info["start"],
                "end": info["end"],
                "token_class": token_class,
                "token_index": info["token_index"],
                "col": edge_idx,
                "is_free": False,
            }
        )
    for edge_idx in lp["word_free_edges"].get(word_idx, []):
        info = lp["free_edge_info"][edge_idx]
        local_edges.append(
            {
                "start": info["start"],
                "end": info["end"],
                "token_class": None,
                "token_index": None,
                "col": num_f + edge_idx,
                "is_free": True,
            }
        )

    local_edges.sort(
        key=lambda edge: (
            edge["start"],
            edge["end"],
            edge["is_free"],
            -1 if edge["token_class"] is None else edge["token_class"],
        )
    )
    pattern_key = (
        lp["word_lengths"][word_idx],
        tuple(
            (
                edge["start"],
                edge["end"],
                -1 if edge["token_class"] is None else edge["token_class"],
                1 if edge["is_free"] else 0,
            )
            for edge in local_edges
        ),
    )
    token_class_to_index = {}
    for edge in local_edges:
        if edge["token_class"] is not None:
            token_class_to_index[edge["token_class"]] = edge["token_index"]
    return local_edges, pattern_key, token_class_to_index


def enumerate_local_pattern_paths(local_edges, target: int, *, max_paths: int):
    by_start = defaultdict(list)
    for local_idx, edge in enumerate(local_edges):
        by_start[edge["start"]].append((edge["end"], local_idx, edge["token_class"]))

    paths = []

    def visit(position: int, local_indices: tuple[int, ...], token_classes: frozenset[int]):
        if len(paths) > max_paths:
            return
        if position == target:
            paths.append((local_indices, token_classes))
            return
        for end, local_idx, token_class in by_start.get(position, []):
            next_classes = token_classes if token_class is None else token_classes | frozenset((token_class,))
            visit(end, (*local_indices, local_idx), next_classes)

    visit(0, tuple(), frozenset())
    if len(paths) > max_paths:
        return None
    return paths


def word_support_cut_from_paths(
    lp,
    f_values,
    g_values,
    t_values,
    word_idx: int,
    selected_tokens: tuple[int, ...],
    paths,
    *,
    tolerance: float,
):
    num_f = lp["num_nonfree_edges"]
    selected_set = set(selected_tokens)
    word_columns = []
    current_values = []
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        word_columns.append(edge_idx)
        current_values.append(float(f_values[edge_idx]))
    for edge_idx in lp["word_free_edges"].get(word_idx, []):
        word_columns.append(num_f + edge_idx)
        current_values.append(float(g_values[edge_idx]))
    col_position = {col_idx: idx for idx, col_idx in enumerate(word_columns)}

    num_dual_vars = len(word_columns) + 1
    rows = []
    cols = []
    data = []
    rhs = []
    for row_idx, (path_columns, path_tokens) in enumerate(paths):
        for col_idx in path_columns:
            rows.append(row_idx)
            cols.append(col_position[col_idx])
            data.append(1.0)
        rows.append(row_idx)
        cols.append(len(word_columns))
        data.append(1.0)
        rhs.append(float(len(path_tokens & selected_set)))

    constraints = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(paths), num_dual_vars),
        dtype=float,
    ).tocsr()
    objective = -np.array([*current_values, 1.0], dtype=float)
    result = linprog(
        c=objective,
        A_ub=constraints,
        b_ub=np.array(rhs, dtype=float),
        bounds=[(None, None)] * num_dual_vars,
        method="highs",
    )
    if not result.success:
        return None

    dual_value = -float(result.fun)
    support_value = float(t_values[list(selected_tokens)].sum())
    violation = dual_value - support_value
    if violation <= tolerance:
        return None

    edge_coefficients = {
        col_idx: float(coefficient)
        for col_idx, coefficient in zip(word_columns, result.x[: len(word_columns)])
        if abs(coefficient) > 1e-10
    }
    token_coefficients = {token_idx: -1.0 for token_idx in selected_tokens}
    gamma = float(result.x[-1])
    rhs_value = -gamma
    return violation, edge_coefficients, token_coefficients, rhs_value, gamma


def word_hull_cut_from_paths(
    lp,
    f_values,
    g_values,
    t_values,
    word_idx: int,
    selected_tokens: tuple[int, ...],
    paths,
    *,
    tolerance: float,
):
    num_f = lp["num_nonfree_edges"]
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    word_columns = []
    current_values = []
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        word_columns.append(edge_idx)
        current_values.append(float(f_values[edge_idx]))
    for edge_idx in lp["word_free_edges"].get(word_idx, []):
        word_columns.append(num_f + edge_idx)
        current_values.append(float(g_values[edge_idx]))
    col_position = {col_idx: idx for idx, col_idx in enumerate(word_columns)}

    num_edge_vars = len(word_columns)
    gamma_col = num_edge_vars
    token_offset = num_edge_vars + 1
    num_vars = token_offset + len(selected_tokens)

    rows = []
    cols = []
    data = []
    rhs = []
    for row_idx, (path_columns, path_tokens) in enumerate(paths):
        for col_idx in path_columns:
            rows.append(row_idx)
            cols.append(col_position[col_idx])
            data.append(1.0)
        rows.append(row_idx)
        cols.append(gamma_col)
        data.append(1.0)
        for token_idx in path_tokens:
            pos = selected_position.get(token_idx)
            if pos is None:
                continue
            rows.append(row_idx)
            cols.append(token_offset + pos)
            data.append(-1.0)
        rhs.append(0.0)

    sum_row = len(paths)
    for pos in range(len(selected_tokens)):
        rows.append(sum_row)
        cols.append(token_offset + pos)
        data.append(1.0)
    rhs.append(1.0)

    constraints = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(paths) + 1, num_vars),
        dtype=float,
    ).tocsr()
    objective = -np.array(
        [
            *current_values,
            1.0,
            *[-float(t_values[token_idx]) for token_idx in selected_tokens],
        ],
        dtype=float,
    )
    bounds = [(None, None)] * (num_edge_vars + 1) + [(0.0, None)] * len(selected_tokens)
    result = linprog(
        c=objective,
        A_ub=constraints,
        b_ub=np.array(rhs, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None

    violation = -float(result.fun)
    if violation <= tolerance:
        return None

    edge_coefficients = {
        col_idx: float(coefficient)
        for col_idx, coefficient in zip(word_columns, result.x[:num_edge_vars])
        if abs(coefficient) > 1e-10
    }
    token_coefficients = {
        token_idx: -float(coefficient)
        for token_idx, coefficient in zip(selected_tokens, result.x[token_offset:])
        if abs(coefficient) > 1e-10
    }
    gamma = float(result.x[gamma_col])
    rhs_value = -gamma
    coefficient_key = (
        round(gamma, 8),
        tuple(round(float(value), 8) for value in result.x[token_offset:]),
    )
    return violation, edge_coefficients, token_coefficients, rhs_value, coefficient_key


def full_upward_hull_cut_from_paths(
    lp,
    f_values,
    g_values,
    t_values,
    word_idx: int,
    selected_tokens: tuple[int, ...],
    paths,
    *,
    tolerance: float,
):
    num_f = lp["num_nonfree_edges"]
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    word_columns = []
    current_values = []
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        word_columns.append(edge_idx)
        current_values.append(float(f_values[edge_idx]))
    for edge_idx in lp["word_free_edges"].get(word_idx, []):
        word_columns.append(num_f + edge_idx)
        current_values.append(float(g_values[edge_idx]))

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

    for row_idx, (path_columns, path_tokens) in enumerate(paths):
        for col_idx in path_columns:
            pos = col_position[col_idx]
            rows.extend((row_idx, row_idx))
            cols.extend((a_pos_offset + pos, a_neg_offset + pos))
            data.extend((1.0, -1.0))
        for pos in range(num_token_vars):
            rows.append(row_idx)
            cols.append(b_pos_offset + pos)
            data.append(1.0)
        for token_idx in path_tokens:
            pos = selected_position.get(token_idx)
            if pos is None:
                continue
            rows.append(row_idx)
            cols.append(b_neg_offset + pos)
            data.append(-1.0)
        rows.append(row_idx)
        cols.append(gamma_col)
        data.append(1.0)
        rhs.append(0.0)

    norm_row = len(paths)
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
        shape=(len(paths) + 1, num_vars),
        dtype=float,
    ).tocsr()
    objective = np.zeros(num_vars, dtype=float)
    for pos, value in enumerate(current_values):
        objective[a_pos_offset + pos] = -value
        objective[a_neg_offset + pos] = value
    for pos, token_idx in enumerate(selected_tokens):
        token_value = float(t_values[token_idx])
        objective[b_pos_offset + pos] = -token_value
        objective[b_neg_offset + pos] = token_value
    objective[gamma_col] = -1.0

    bounds = [(0.0, None)] * num_vars
    bounds[gamma_col] = (None, None)
    result = linprog(
        c=objective,
        A_ub=constraints,
        b_ub=np.array(rhs, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None

    violation = -float(result.fun)
    if violation <= tolerance:
        return None

    edge_coefficients = {}
    for col_idx, pos in col_position.items():
        coefficient = float(result.x[a_pos_offset + pos] - result.x[a_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            edge_coefficients[col_idx] = coefficient
    token_coefficients = {}
    for token_idx, pos in selected_position.items():
        coefficient = float(result.x[b_pos_offset + pos] - result.x[b_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            token_coefficients[token_idx] = coefficient

    gamma = float(result.x[gamma_col])
    coefficient_key = (
        round(gamma, 8),
        tuple(round(float(token_coefficients.get(token_idx, 0.0)), 8) for token_idx in selected_tokens),
    )
    return violation, edge_coefficients, token_coefficients, -gamma, coefficient_key


def pair_full_upward_hull_cut_from_paths(
    lp,
    f_values,
    g_values,
    t_values,
    left_word_idx: int,
    right_word_idx: int,
    selected_tokens: tuple[int, ...],
    left_paths,
    right_paths,
    *,
    tolerance: float,
):
    num_f = lp["num_nonfree_edges"]
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    selected_set = set(selected_tokens)
    word_columns = []
    current_values = []
    for word_idx in (left_word_idx, right_word_idx):
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            word_columns.append(edge_idx)
            current_values.append(float(f_values[edge_idx]))
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            word_columns.append(num_f + edge_idx)
            current_values.append(float(g_values[edge_idx]))

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
    for left_columns, left_tokens in left_paths:
        left_required = set(left_tokens) & selected_set
        for right_columns, right_tokens in right_paths:
            for col_idx in (*left_columns, *right_columns):
                pos = col_position[col_idx]
                rows.extend((row_idx, row_idx))
                cols.extend((a_pos_offset + pos, a_neg_offset + pos))
                data.extend((1.0, -1.0))
            for pos in range(num_token_vars):
                rows.append(row_idx)
                cols.append(b_pos_offset + pos)
                data.append(1.0)
            for token_idx in left_required | (set(right_tokens) & selected_set):
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
        token_value = float(t_values[token_idx])
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
        return None

    violation = -float(result.fun)
    if violation <= tolerance:
        return None

    edge_coefficients = {}
    for col_idx, pos in col_position.items():
        coefficient = float(result.x[a_pos_offset + pos] - result.x[a_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            edge_coefficients[col_idx] = coefficient
    token_coefficients = {}
    for token_idx, pos in selected_position.items():
        coefficient = float(result.x[b_pos_offset + pos] - result.x[b_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            token_coefficients[token_idx] = coefficient

    gamma = float(result.x[gamma_col])
    coefficient_key = (
        round(gamma, 8),
        tuple(round(float(token_coefficients.get(token_idx, 0.0)), 8) for token_idx in selected_tokens),
    )
    return (
        violation,
        edge_coefficients,
        token_coefficients,
        -gamma,
        coefficient_key,
        build_seconds,
        solve_seconds,
    )


def pair_reduced_fractional_edge_hull_cut_from_paths(
    lp,
    f_values,
    g_values,
    t_values,
    left_word_idx: int,
    right_word_idx: int,
    selected_tokens: tuple[int, ...],
    left_paths,
    right_paths,
    *,
    tolerance: float,
    max_pair_rows: int,
):
    num_f = lp["num_nonfree_edges"]
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    selected_set = set(selected_tokens)
    word_columns = []
    current_values = []
    for word_idx in (left_word_idx, right_word_idx):
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            value = float(f_values[edge_idx])
            if tolerance < value < 1.0 - tolerance:
                word_columns.append(edge_idx)
                current_values.append(value)
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            value = float(g_values[edge_idx])
            if tolerance < value < 1.0 - tolerance:
                word_columns.append(num_f + edge_idx)
                current_values.append(value)

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

    left_signatures = projected_pair_hull_path_signatures(
        left_paths,
        selected_set,
        selected_position,
        col_position,
    )
    right_signatures = projected_pair_hull_path_signatures(
        right_paths,
        selected_set,
        selected_position,
        col_position,
    )
    pair_row_keys = []
    pair_row_key_set = set()
    limit_hit = False
    for left_kept, left_mask in left_signatures:
        for right_kept, right_mask in right_signatures:
            row_key = (left_kept + right_kept, left_mask | right_mask)
            if row_key in pair_row_key_set:
                continue
            if max_pair_rows > 0 and len(pair_row_keys) >= max_pair_rows:
                limit_hit = True
                break
            pair_row_key_set.add(row_key)
            pair_row_keys.append(row_key)
        if limit_hit:
            break

    rows = []
    cols = []
    data = []
    rhs = []
    for row_idx, (kept_positions, token_mask) in enumerate(pair_row_keys):
        for pos in kept_positions:
            rows.extend((row_idx, row_idx))
            cols.extend((a_pos_offset + pos, a_neg_offset + pos))
            data.extend((1.0, -1.0))
        for pos in range(num_token_vars):
            rows.append(row_idx)
            cols.append(b_pos_offset + pos)
            data.append(1.0)
            if token_mask & (1 << pos):
                rows.append(row_idx)
                cols.append(b_neg_offset + pos)
                data.append(-1.0)
        rows.append(row_idx)
        cols.append(gamma_col)
        data.append(1.0)
        rhs.append(0.0)

    row_idx = len(pair_row_keys)

    if limit_hit:
        return {
            "skip_reason": "reduced_rows",
            "build_seconds": time.monotonic() - build_start,
            "solve_seconds": 0.0,
            "validation_seconds": 0.0,
            "reduced_rows": row_idx,
            "edge_vars": num_edge_vars,
            "left_signatures": len(left_signatures),
            "right_signatures": len(right_signatures),
        }

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
        token_value = float(t_values[token_idx])
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
    base = {
        "skip_reason": None,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "validation_seconds": 0.0,
        "reduced_rows": row_idx,
        "edge_vars": num_edge_vars,
        "left_signatures": len(left_signatures),
        "right_signatures": len(right_signatures),
    }
    if not result.success:
        base["skip_reason"] = "linprog"
        return base

    reduced_violation = -float(result.fun)
    if reduced_violation <= tolerance:
        return base

    edge_coefficients = {}
    for col_idx, pos in col_position.items():
        coefficient = float(result.x[a_pos_offset + pos] - result.x[a_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            edge_coefficients[int(col_idx)] = coefficient
    token_coefficients = {}
    for token_idx, pos in selected_position.items():
        coefficient = float(result.x[b_pos_offset + pos] - result.x[b_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            token_coefficients[int(token_idx)] = coefficient
    rhs_value = -float(result.x[gamma_col])

    validation_start = time.monotonic()
    max_slack = pair_hull_max_projected_row_slack(
        edge_coefficients,
        token_coefficients,
        rhs_value,
        selected_tokens,
        pair_row_keys,
        word_columns,
    )
    current_violation = pair_hull_current_violation(
        edge_coefficients,
        token_coefficients,
        rhs_value,
        selected_tokens,
        f_values,
        g_values,
        t_values,
        num_f,
    )
    base["validation_seconds"] = time.monotonic() - validation_start
    if max_slack > 1e-7 or current_violation <= tolerance:
        base["skip_reason"] = "invalid_reduced_cut"
        return base

    coefficient_key = (
        "fractional_edges_shared_colors",
        round(-rhs_value, 8),
        tuple(round(float(token_coefficients.get(token_idx, 0.0)), 8) for token_idx in selected_tokens),
        tuple(sorted((int(col_idx), round(float(coefficient), 8)) for col_idx, coefficient in edge_coefficients.items())),
    )
    base.update(
        {
            "violation": float(current_violation),
            "edge_coefficients": edge_coefficients,
            "token_coefficients": token_coefficients,
            "rhs": rhs_value,
            "coefficient_key": coefficient_key,
        }
    )
    return base


def triple_reduced_fractional_edge_hull_cut_from_paths(
    lp,
    f_values,
    g_values,
    t_values,
    word_indices: tuple[int, int, int],
    selected_tokens: tuple[int, ...],
    paths_list,
    *,
    tolerance: float,
    max_rows: int,
):
    num_f = lp["num_nonfree_edges"]
    selected_position = {token_idx: idx for idx, token_idx in enumerate(selected_tokens)}
    selected_set = set(selected_tokens)
    word_columns = []
    current_values = []
    for word_idx in word_indices:
        for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
            value = float(f_values[edge_idx])
            if tolerance < value < 1.0 - tolerance:
                word_columns.append(int(edge_idx))
                current_values.append(value)
        for edge_idx in lp["word_free_edges"].get(word_idx, []):
            value = float(g_values[edge_idx])
            if tolerance < value < 1.0 - tolerance:
                word_columns.append(int(num_f + edge_idx))
                current_values.append(value)

    build_start = time.monotonic()
    col_position = {col_idx: idx for idx, col_idx in enumerate(word_columns)}
    signatures = [
        projected_pair_hull_path_signatures(paths, selected_set, selected_position, col_position)
        for paths in paths_list
    ]
    signatures.sort(key=len)
    pair_keys = set()
    for first_edges, first_mask in signatures[0]:
        for second_edges, second_mask in signatures[1]:
            pair_keys.add((first_edges + second_edges, first_mask | second_mask))
    row_keys = []
    row_key_set = set()
    limit_hit = False
    for pair_edges, pair_mask in pair_keys:
        for third_edges, third_mask in signatures[2]:
            row_key = (pair_edges + third_edges, pair_mask | third_mask)
            if row_key in row_key_set:
                continue
            if max_rows > 0 and len(row_keys) >= max_rows:
                limit_hit = True
                break
            row_key_set.add(row_key)
            row_keys.append(row_key)
        if limit_hit:
            break

    num_edge_vars = len(word_columns)
    num_token_vars = len(selected_tokens)
    a_pos_offset = 0
    a_neg_offset = num_edge_vars
    b_pos_offset = 2 * num_edge_vars
    b_neg_offset = b_pos_offset + num_token_vars
    gamma_col = b_neg_offset + num_token_vars
    num_vars = gamma_col + 1

    if limit_hit:
        return {
            "skip_reason": "reduced_rows",
            "build_seconds": time.monotonic() - build_start,
            "solve_seconds": 0.0,
            "validation_seconds": 0.0,
            "reduced_rows": len(row_keys),
            "edge_vars": num_edge_vars,
        }

    rows = []
    cols = []
    data = []
    rhs = []
    for row_idx, (kept_positions, token_mask) in enumerate(row_keys):
        for pos in kept_positions:
            rows.extend((row_idx, row_idx))
            cols.extend((a_pos_offset + pos, a_neg_offset + pos))
            data.extend((1.0, -1.0))
        for pos in range(num_token_vars):
            rows.append(row_idx)
            cols.append(b_pos_offset + pos)
            data.append(1.0)
            if token_mask & (1 << pos):
                rows.append(row_idx)
                cols.append(b_neg_offset + pos)
                data.append(-1.0)
        rows.append(row_idx)
        cols.append(gamma_col)
        data.append(1.0)
        rhs.append(0.0)

    norm_row = len(row_keys)
    for pos in range(num_edge_vars):
        rows.extend((norm_row, norm_row))
        cols.extend((a_pos_offset + pos, a_neg_offset + pos))
        data.extend((1.0, 1.0))
    for pos in range(num_token_vars):
        rows.extend((norm_row, norm_row))
        cols.extend((b_pos_offset + pos, b_neg_offset + pos))
        data.extend((1.0, 1.0))
    rhs.append(1.0)

    constraints = sp.coo_matrix((data, (rows, cols)), shape=(norm_row + 1, num_vars), dtype=float).tocsr()
    objective = np.zeros(num_vars, dtype=float)
    for pos, value in enumerate(current_values):
        objective[a_pos_offset + pos] = -value
        objective[a_neg_offset + pos] = value
    for pos, token_idx in enumerate(selected_tokens):
        token_value = float(t_values[token_idx])
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
    base = {
        "skip_reason": None,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "validation_seconds": 0.0,
        "reduced_rows": len(row_keys),
        "edge_vars": num_edge_vars,
    }
    if not result.success:
        base["skip_reason"] = "linprog"
        return base

    reduced_violation = -float(result.fun)
    if reduced_violation <= tolerance:
        return base

    edge_coefficients = {}
    for col_idx, pos in col_position.items():
        coefficient = float(result.x[a_pos_offset + pos] - result.x[a_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            edge_coefficients[int(col_idx)] = coefficient
    token_coefficients = {}
    for token_idx, pos in selected_position.items():
        coefficient = float(result.x[b_pos_offset + pos] - result.x[b_neg_offset + pos])
        if abs(coefficient) > 1e-10:
            token_coefficients[int(token_idx)] = coefficient
    rhs_value = -float(result.x[gamma_col])

    validation_start = time.monotonic()
    max_slack = pair_hull_max_projected_row_slack(
        edge_coefficients,
        token_coefficients,
        rhs_value,
        selected_tokens,
        row_keys,
        word_columns,
    )
    current_violation = pair_hull_current_violation(
        edge_coefficients,
        token_coefficients,
        rhs_value,
        selected_tokens,
        f_values,
        g_values,
        t_values,
        num_f,
    )
    base["validation_seconds"] = time.monotonic() - validation_start
    if max_slack > 1e-7 or current_violation <= tolerance:
        base["skip_reason"] = "invalid_reduced_cut"
        return base

    coefficient_key = (
        "fractional_edges_shared_colors",
        round(-rhs_value, 8),
        tuple(round(float(token_coefficients.get(token_idx, 0.0)), 8) for token_idx in selected_tokens),
        tuple(sorted((int(col_idx), round(float(coefficient), 8)) for col_idx, coefficient in edge_coefficients.items())),
    )
    base.update(
        {
            "violation": float(current_violation),
            "edge_coefficients": edge_coefficients,
            "token_coefficients": token_coefficients,
            "rhs": rhs_value,
            "coefficient_key": coefficient_key,
        }
    )
    return base


def projected_pair_hull_path_signatures(paths, selected_set, selected_position, col_position):
    signatures = set()
    for path_columns, path_tokens in paths:
        kept_positions = tuple(sorted(col_position[col_idx] for col_idx in path_columns if col_idx in col_position))
        token_mask = 0
        for token_idx in set(path_tokens) & selected_set:
            token_mask |= 1 << selected_position[token_idx]
        signatures.add((kept_positions, token_mask))
    return tuple(signatures)


def pair_hull_max_projected_row_slack(edge_coefficients, token_coefficients, rhs, selected_tokens, pair_row_keys, word_columns):
    positive_token_sum = sum(max(0.0, float(token_coefficients.get(token_idx, 0.0))) for token_idx in selected_tokens)
    max_slack = -float("inf")
    for kept_positions, token_mask in pair_row_keys:
        lhs = positive_token_sum
        for pos in kept_positions:
            lhs += float(edge_coefficients.get(int(word_columns[pos]), 0.0))
        for pos, token_idx in enumerate(selected_tokens):
            if token_mask & (1 << pos):
                lhs += min(0.0, float(token_coefficients.get(token_idx, 0.0)))
        max_slack = max(max_slack, lhs - rhs)
    return max_slack


def pair_hull_current_violation(edge_coefficients, token_coefficients, rhs, selected_tokens, f_values, g_values, t_values, num_f):
    lhs = 0.0
    for col_idx, coefficient in edge_coefficients.items():
        if col_idx < num_f:
            lhs += coefficient * float(f_values[col_idx])
        else:
            lhs += coefficient * float(g_values[col_idx - num_f])
    for token_idx in selected_tokens:
        lhs += float(token_coefficients.get(token_idx, 0.0)) * float(t_values[token_idx])
    return lhs - rhs


def small_quantile_summary(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "max": float(np.max(arr)),
    }


def separate_bad_vocab_escape_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 800,
    max_paths: int = 50000,
    max_exact_hitting_tokens: int = 80,
):
    """Separate cuts from a nearby rounded vocabulary's bad word costs.

    Let V be the top-sum(t) rounded vocabulary. If V tokenizes word w in K
    tokens, every path with <K tokens must use at least one token outside V.
    For any hitting set H of those better paths:

        word_cost >= K * (1 - sum_{tau in H} t_tau)

    The better paths are enumerated exactly up to K-1 tokens; otherwise no cut
    is emitted for that word.
    """

    budget = int(round(float(t_values.sum())))
    if budget <= 0:
        return []
    active_tokens = rounded_active_token_set(t_values, budget)
    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    candidate_rows = []
    for word_idx in rank_suspicious_words(lp, f_values, max_words=max_words):
        if lp["word_weights"][word_idx] <= 1:
            continue
        current_cost = float(
            f_values[lp["word_nonfree_edges"].get(word_idx, [])].sum()
            + g_values[lp["word_free_edges"].get(word_idx, [])].sum()
        )
        rounded_cost = word_shortest_path_cost_active(lp, word_idx, active_tokens)
        gap = rounded_cost - current_cost
        if gap > tolerance and rounded_cost >= 2:
            candidate_rows.append((gap * lp["word_weights"][word_idx], gap, word_idx, rounded_cost, current_cost))
    candidate_rows.sort(reverse=True)

    for _, gap, word_idx, rounded_cost, current_cost in candidate_rows:
        full_key_prefix = ("bad_vocab_escape", word_idx, rounded_cost)
        if any(key[:3] == full_key_prefix for key in existing_cut_keys):
            continue
        better_path_sets = enumerate_better_path_escape_sets(
            lp,
            word_idx,
            active_tokens,
            max_length=rounded_cost - 1,
            max_paths=max_paths,
        )
        if better_path_sets is None or not better_path_sets:
            continue
        if any(len(token_set) == 0 for token_set in better_path_sets):
            continue

        token_universe = sorted(set().union(*better_path_sets))
        if len(token_universe) <= max_exact_hitting_tokens:
            hitting_set = exact_min_weight_hitting_set(better_path_sets, t_values, token_universe)
        else:
            hitting_set = greedy_weighted_hitting_set(better_path_sets, t_values)
        if not hitting_set:
            continue
        if not all(hitting_set & set(token_set) for token_set in better_path_sets):
            continue

        hit_sum = float(t_values[list(hitting_set)].sum())
        lhs_value = current_cost + rounded_cost * hit_sum
        rhs_value = float(rounded_cost)
        violation = rhs_value - lhs_value
        if violation <= tolerance:
            continue

        full_key = (*full_key_prefix, tuple(sorted(hitting_set)))
        if full_key in existing_cut_keys:
            continue

        entries = []
        entries.extend((edge_idx, -1.0) for edge_idx in lp["word_nonfree_edges"].get(word_idx, []))
        entries.extend((num_f + edge_idx, -1.0) for edge_idx in lp["word_free_edges"].get(word_idx, []))
        entries.extend((t_offset + token_idx, -float(rounded_cost)) for token_idx in hitting_set)
        violations.append((violation, full_key, entries, -rhs_value))

    return violations


def separate_bad_vocab_improvement_cut_specs(
    lp,
    f_values,
    g_values,
    t_values,
    *,
    existing_cut_keys: set[tuple],
    tolerance: float,
    max_words: int = 800,
    max_paths: int = 50000,
    max_escape_tokens: int = 18,
):
    """Separate bad-vocabulary improvement-capacity cuts.

    For rounded vocabulary V and word cost K under V, every shorter path has an
    off-vocabulary escape-token set E and improvement K-|p|. For all active
    escape sets U, cap(U) is the best improvement achievable by any enumerated
    path with E subset U. A modular upper bound on cap gives:

        K - word_cost <= alpha + sum beta_tau t_tau
    """

    budget = int(round(float(t_values.sum())))
    if budget <= 0:
        return []
    active_tokens = rounded_active_token_set(t_values, budget)
    violations = []
    num_f = lp["num_nonfree_edges"]
    num_g = lp["num_free_edges"]
    t_offset = num_f + num_g

    candidate_rows = []
    for word_idx in rank_suspicious_words(lp, f_values, max_words=max_words):
        if lp["word_weights"][word_idx] <= 1:
            continue
        current_cost = float(
            f_values[lp["word_nonfree_edges"].get(word_idx, [])].sum()
            + g_values[lp["word_free_edges"].get(word_idx, [])].sum()
        )
        rounded_cost = word_shortest_path_cost_active(lp, word_idx, active_tokens)
        improvement = rounded_cost - current_cost
        if improvement > tolerance and rounded_cost >= 2:
            candidate_rows.append(
                (improvement * lp["word_weights"][word_idx], improvement, word_idx, rounded_cost, current_cost)
            )
    candidate_rows.sort(reverse=True)

    for _, improvement, word_idx, rounded_cost, current_cost in candidate_rows:
        full_key_prefix = ("bad_vocab_improvement", word_idx, rounded_cost)
        if any(key[:3] == full_key_prefix for key in existing_cut_keys):
            continue
        better_paths = enumerate_better_path_escape_sets_with_lengths(
            lp,
            word_idx,
            active_tokens,
            max_length=rounded_cost - 1,
            max_paths=max_paths,
        )
        if better_paths is None or not better_paths:
            continue
        if any(len(escape_set) == 0 for escape_set, _ in better_paths):
            continue

        token_universe = sorted(set().union(*(escape_set for escape_set, _ in better_paths)))
        if len(token_universe) > max_escape_tokens:
            continue
        capacities = improvement_capacities(
            better_paths,
            token_universe,
            rounded_cost=rounded_cost,
        )
        bound = modular_upper_bound_for_capacities(capacities, t_values, token_universe)
        if bound is None:
            continue
        alpha, beta_values, rhs_at_current = bound
        violation = improvement - rhs_at_current
        if violation <= tolerance:
            continue

        beta_key = tuple(round(float(beta), 8) for beta in beta_values)
        full_key = (*full_key_prefix, tuple(token_universe), beta_key)
        if full_key in existing_cut_keys:
            continue

        entries = []
        entries.extend((edge_idx, -1.0) for edge_idx in lp["word_nonfree_edges"].get(word_idx, []))
        entries.extend((num_f + edge_idx, -1.0) for edge_idx in lp["word_free_edges"].get(word_idx, []))
        entries.extend(
            (t_offset + token_idx, -float(beta))
            for token_idx, beta in zip(token_universe, beta_values)
            if abs(beta) > 1e-12
        )
        violations.append((violation, full_key, entries, float(alpha - rounded_cost)))

    return violations


def rounded_active_token_set(t_values, budget: int) -> set[int]:
    rows = [(float(value), token_idx) for token_idx, value in enumerate(t_values)]
    rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return {token_idx for _, token_idx in rows[:budget]}


def word_shortest_path_cost_active(lp, word_idx: int, active_tokens: set[int]) -> int:
    by_start = defaultdict(list)
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        info = lp["nonfree_edge_info"][edge_idx]
        if info["token_index"] in active_tokens:
            by_start[info["start"]].append(info["end"])
    for edge_idx in lp["word_free_edges"].get(word_idx, []):
        info = lp["free_edge_info"][edge_idx]
        by_start[info["start"]].append(info["end"])

    target = lp["word_lengths"][word_idx]
    costs = [10**9] * (target + 1)
    costs[0] = 0
    for position in range(target):
        if costs[position] >= 10**9:
            continue
        next_cost = costs[position] + 1
        for end in by_start.get(position, []):
            if next_cost < costs[end]:
                costs[end] = next_cost
    return int(costs[target])


def enumerate_better_path_escape_sets(
    lp,
    word_idx: int,
    active_tokens: set[int],
    *,
    max_length: int,
    max_paths: int,
):
    by_start = defaultdict(list)
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        info = lp["nonfree_edge_info"][edge_idx]
        token_idx = info["token_index"]
        by_start[info["start"]].append((info["end"], None if token_idx in active_tokens else token_idx))
    for edge_idx in lp["word_free_edges"].get(word_idx, []):
        info = lp["free_edge_info"][edge_idx]
        by_start[info["start"]].append((info["end"], None))

    target = lp["word_lengths"][word_idx]
    results = []

    def visit(position: int, remaining: int, escape_set: frozenset[int]):
        if len(results) > max_paths:
            return
        if position == target:
            results.append(escape_set)
            return
        if remaining == 0:
            return
        for end, token_idx in by_start.get(position, []):
            next_set = escape_set if token_idx is None else escape_set | frozenset((token_idx,))
            visit(end, remaining - 1, next_set)

    visit(0, max_length, frozenset())
    if len(results) > max_paths:
        return None
    return results


def enumerate_better_path_escape_sets_with_lengths(
    lp,
    word_idx: int,
    active_tokens: set[int],
    *,
    max_length: int,
    max_paths: int,
):
    by_start = defaultdict(list)
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        info = lp["nonfree_edge_info"][edge_idx]
        token_idx = info["token_index"]
        by_start[info["start"]].append((info["end"], None if token_idx in active_tokens else token_idx))
    for edge_idx in lp["word_free_edges"].get(word_idx, []):
        info = lp["free_edge_info"][edge_idx]
        by_start[info["start"]].append((info["end"], None))

    target = lp["word_lengths"][word_idx]
    results = []

    def visit(position: int, remaining: int, length: int, escape_set: frozenset[int]):
        if len(results) > max_paths:
            return
        if position == target:
            results.append((escape_set, length))
            return
        if remaining == 0:
            return
        for end, token_idx in by_start.get(position, []):
            next_set = escape_set if token_idx is None else escape_set | frozenset((token_idx,))
            visit(end, remaining - 1, length + 1, next_set)

    visit(0, max_length, 0, frozenset())
    if len(results) > max_paths:
        return None
    return results


def improvement_capacities(better_paths, token_universe, *, rounded_cost: int):
    token_position = {token_idx: bit for bit, token_idx in enumerate(token_universe)}
    rank = len(token_universe)
    capacities = {mask: 0.0 for mask in range(1 << rank)}
    for escape_set, path_length in better_paths:
        mask = 0
        for token_idx in escape_set:
            mask |= 1 << token_position[token_idx]
        capacities[mask] = max(capacities[mask], float(rounded_cost - path_length))

    for bit in range(rank):
        bit_value = 1 << bit
        for mask in range(1 << rank):
            if mask & bit_value:
                capacities[mask] = max(capacities[mask], capacities[mask ^ bit_value])
    return capacities


def conflict_vertices_for_word(lp, f_values, g_values, word_idx: int, *, max_vertices: int, tolerance: float):
    num_f = lp["num_nonfree_edges"]
    vertices = []
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        value = float(f_values[edge_idx])
        if value <= tolerance:
            continue
        info = lp["nonfree_edge_info"][edge_idx]
        vertices.append(
            {
                "col": edge_idx,
                "value": value,
                "start": info["start"],
                "end": info["end"],
            }
        )
    for edge_idx in lp["word_free_edges"].get(word_idx, []):
        value = float(g_values[edge_idx])
        if value <= tolerance:
            continue
        info = lp["free_edge_info"][edge_idx]
        vertices.append(
            {
                "col": num_f + edge_idx,
                "value": value,
                "start": info["start"],
                "end": info["end"],
            }
        )
    vertices.sort(key=lambda item: (min(item["value"], 1.0 - item["value"]), item["value"]), reverse=True)
    return vertices[:max_vertices]


def build_interval_conflict_adjacency(vertices):
    adjacency = [set() for _ in vertices]
    for left in range(len(vertices)):
        for right in range(left + 1, len(vertices)):
            if intervals_overlap(
                vertices[left]["start"],
                vertices[left]["end"],
                vertices[right]["start"],
                vertices[right]["end"],
            ):
                adjacency[left].add(right)
                adjacency[right].add(left)
    return adjacency


def find_odd_cycles(adjacency, *, max_cycle_len: int):
    seen = set()
    cycles = []
    for start in range(len(adjacency)):
        stack = [(start, [start], {start})]
        while stack:
            current, path, used = stack.pop()
            if len(path) > max_cycle_len:
                continue
            for nxt in adjacency[current]:
                if nxt == start and len(path) >= 5 and len(path) % 2 == 1:
                    cycle_key = canonical_cycle_key(path)
                    if cycle_key not in seen:
                        seen.add(cycle_key)
                        cycles.append(tuple(path))
                    continue
                if nxt <= start or nxt in used or len(path) == max_cycle_len:
                    continue
                stack.append((nxt, [*path, nxt], used | {nxt}))
    return cycles


def canonical_cycle_key(cycle):
    forward = list(cycle)
    reverse = [cycle[0], *reversed(cycle[1:])]
    rotations = []
    for values in (forward, reverse):
        for idx in range(len(values)):
            rotations.append(tuple(values[idx:] + values[:idx]))
    return min(rotations)


def group_current_cost(lp, selected_words, f_values, g_values) -> float:
    total = 0.0
    for word_idx in selected_words:
        word_weight = float(lp["word_weights"][word_idx])
        nonfree_edges = lp["word_nonfree_edges"].get(word_idx, [])
        free_edges = lp["word_free_edges"].get(word_idx, [])
        total += word_weight * float(f_values[nonfree_edges].sum() + g_values[free_edges].sum())
    return total


def group_value_function(lp, selected_words, selected_tokens):
    token_position = {token_idx: bit for bit, token_idx in enumerate(selected_tokens)}
    values = {mask: 0.0 for mask in range(1 << len(selected_tokens))}
    for mask in values:
        for word_idx in selected_words:
            values[mask] += float(lp["word_weights"][word_idx]) * word_shortest_path_cost(
                lp,
                word_idx,
                token_position,
                mask,
            )
    return values


def group_budget_value_function(lp, selected_words, selected_tokens):
    value_by_mask = group_value_function(lp, selected_words, selected_tokens)
    rank = len(selected_tokens)
    values_by_budget = {budget: float("inf") for budget in range(rank + 1)}
    for mask, value in value_by_mask.items():
        active = int(mask.bit_count())
        if value < values_by_budget[active]:
            values_by_budget[active] = float(value)

    best_so_far = float("inf")
    for budget in range(rank + 1):
        best_so_far = min(best_so_far, values_by_budget[budget])
        values_by_budget[budget] = best_so_far
    return values_by_budget


def word_shortest_path_cost(lp, word_idx: int, token_position: dict[int, int], mask: int) -> float:
    by_start = defaultdict(list)
    for edge_idx in lp["word_nonfree_edges"].get(word_idx, []):
        info = lp["nonfree_edge_info"][edge_idx]
        token_idx = info["token_index"]
        bit = token_position.get(token_idx)
        if bit is None or (mask & (1 << bit)):
            by_start[info["start"]].append(info["end"])
    for edge_idx in lp["word_free_edges"].get(word_idx, []):
        info = lp["free_edge_info"][edge_idx]
        by_start[info["start"]].append(info["end"])

    target = lp["word_lengths"][word_idx]
    costs = [float("inf")] * (target + 1)
    costs[0] = 0.0
    for position in range(target):
        if not np.isfinite(costs[position]):
            continue
        next_cost = costs[position] + 1.0
        for end in by_start.get(position, []):
            if end <= target and next_cost < costs[end]:
                costs[end] = next_cost
    return float(costs[target])


def affine_lower_bound_for_values(values_by_mask, t_values, selected_tokens):
    rank = len(selected_tokens)
    objective = np.array([-1.0, *[-float(t_values[token_idx]) for token_idx in selected_tokens]])
    a_ub = []
    b_ub = []
    for mask, value in values_by_mask.items():
        row = [1.0]
        for bit in range(rank):
            row.append(1.0 if mask & (1 << bit) else 0.0)
        a_ub.append(row)
        b_ub.append(float(value))

    result = linprog(
        c=objective,
        A_ub=np.array(a_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        bounds=[(None, None)] * (rank + 1),
        method="highs",
    )
    if not result.success:
        return None
    alpha = float(result.x[0])
    beta_values = np.array(result.x[1:], dtype=float)
    rhs_at_current = -float(result.fun)
    return alpha, beta_values, rhs_at_current


def affine_lower_bound_for_budget_values(values_by_budget, t_values, selected_tokens):
    current_budget = float(sum(float(t_values[token_idx]) for token_idx in selected_tokens))
    objective = np.array([-1.0, -current_budget], dtype=float)
    a_ub = []
    b_ub = []
    for budget, value in values_by_budget.items():
        row = [1.0, float(budget)]
        a_ub.append(row)
        b_ub.append(float(value))

    result = linprog(
        c=objective,
        A_ub=np.array(a_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        bounds=[(None, None), (None, None)],
        method="highs",
    )
    if not result.success:
        return None
    alpha = float(result.x[0])
    beta = float(result.x[1])
    rhs_at_current = -float(result.fun)
    return alpha, beta, rhs_at_current


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


def exact_min_weight_hitting_set(path_token_sets, t_values, token_universe):
    from scipy.optimize import Bounds, LinearConstraint, milp

    token_position = {token_idx: idx for idx, token_idx in enumerate(token_universe)}
    rows = []
    cols = []
    data = []
    for row_idx, token_set in enumerate(path_token_sets):
        for token_idx in token_set:
            rows.append(row_idx)
            cols.append(token_position[token_idx])
            data.append(1.0)

    if not rows:
        return set()
    constraint_matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(path_token_sets), len(token_universe)),
        dtype=float,
    ).tocsr()
    constraints = LinearConstraint(
        constraint_matrix,
        lb=np.ones(len(path_token_sets), dtype=float),
        ub=np.full(len(path_token_sets), np.inf, dtype=float),
    )
    objective = np.array([float(t_values[token_idx]) for token_idx in token_universe], dtype=float)
    result = milp(
        c=objective,
        integrality=np.ones(len(token_universe), dtype=int),
        bounds=Bounds(np.zeros(len(token_universe)), np.ones(len(token_universe))),
        constraints=constraints,
        options={"disp": False, "time_limit": 2.0},
    )
    if not result.success:
        return set()
    return {
        token_idx
        for token_idx, value in zip(token_universe, result.x)
        if value > 0.5
    }


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
