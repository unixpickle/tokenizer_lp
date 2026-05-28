from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from tokenisation_lp.corpus import load_texts
from tokenisation_lp.dp_tokenizer import LpDpTokenizer
from tokenisation_lp.evaluation import evaluate_texts
from tokenisation_lp.lp_training import (
    build_standard_form,
    byte_level_alphabet,
    candidates_from_solution,
    count_pretokenized_strings,
    prepare_lp_data,
)
from tokenisation_lp.pretokenization import DEFAULT_SPECIAL_TOKENS, DEFAULT_UNK_TOKEN, build_pretokenizer


LOGGER = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    try:
        import cma
    except ImportError as exc:
        raise SystemExit("Missing optional dependency 'cma'. Run with: uv run --with cma tokenizer-lp-cma-rounding ...") from exc

    start = time.monotonic()
    state_dir = resolve_state_dir(args)
    state = load_state(state_dir)
    solution = np.load(resolve_solution_path(args, state_dir), allow_pickle=False)

    LOGGER.info("Loading corpus and rebuilding LP")
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
    expected_vars = lp["num_nonfree_edges"] + lp["num_free_edges"] + lp["num_tokens"]
    if solution.shape != (expected_vars,):
        raise ValueError(f"solution shape {solution.shape} does not match rebuilt LP vars {expected_vars}")

    base_vocab, excluded = build_base_vocab()
    budget = args.vocab_size - len(base_vocab)
    if budget <= 0:
        raise ValueError(f"vocab_size={args.vocab_size} leaves non-positive LP token budget {budget}")

    all_candidates = [token for token in candidates_from_solution(tokens, lp, solution) if token.token not in excluded]
    ordered = sorted(
        all_candidates,
        key=lambda token: (token.lp_value, token.token_instance_count, len(token.token)),
        reverse=True,
    )
    if len(ordered) < budget:
        raise ValueError(f"Only {len(ordered)} positive LP candidates for budget {budget}")

    fixed_in, search_candidates, local_budget = split_search_window(ordered, budget, args.candidate_window)
    x0 = np.asarray([token.lp_value for token in search_candidates], dtype=float)
    deterministic_indices = select_from_vector(search_candidates, local_budget, x0)

    cache: dict[tuple[int, ...], int] = {}
    best = SearchBest(tokens=math.inf, eval=-1, indices=tuple(), source="")
    eval_count = 0

    def evaluate_indices(indices: tuple[int, ...], source: str) -> int:
        nonlocal best, eval_count
        indices = tuple(indices)
        cached = cache.get(indices)
        if cached is not None:
            return cached
        selected = [*fixed_in, *(search_candidates[idx] for idx in indices)]
        tokenizer = LpDpTokenizer(
            [*base_vocab, *(token.token for token in selected)],
            pretokenizer_mode=args.pretokenizer,
            unk_token=DEFAULT_UNK_TOKEN,
        )
        stats = evaluate_texts(source, tokenizer, texts, num_workers=args.eval_workers)
        eval_count += 1
        cache[indices] = int(stats.tokens)
        if stats.tokens < best.tokens:
            best = SearchBest(tokens=int(stats.tokens), eval=eval_count, indices=indices, source=source)
            payload = best_payload(best, ordered[:budget], fixed_in, search_candidates, state)
            LOGGER.info("NEW_BEST %s", json.dumps(compact_payload(payload), ensure_ascii=False, sort_keys=True))
            write_outputs(args, payload, base_vocab, fixed_in, search_candidates)
        return int(stats.tokens)

    baseline_tokens = evaluate_indices(deterministic_indices, "deterministic_lp")
    LOGGER.info(
        "Starting CMA rounding search: baseline=%d fixed=%d search_dim=%d local_budget=%d sigma=%.6g popsize=%d",
        baseline_tokens,
        len(fixed_in),
        len(search_candidates),
        local_budget,
        args.sigma,
        args.popsize,
    )

    options = {
        "popsize": args.popsize,
        "seed": args.seed,
        "maxfevals": args.max_evals,
        "verbose": -9,
        "bounds": [args.lower_bound, args.upper_bound],
    }
    es = cma.CMAEvolutionStrategy(x0, args.sigma, options)
    generation = 0
    deadline = start + args.time_limit_seconds if args.time_limit_seconds > 0 else None
    while not es.stop() and eval_count < args.max_evals:
        if deadline is not None and time.monotonic() >= deadline:
            break
        solutions = es.ask()
        fitnesses = []
        used_solutions = []
        for vector in solutions:
            if deadline is not None and time.monotonic() >= deadline:
                break
            indices = select_from_vector(search_candidates, local_budget, vector)
            fitnesses.append(float(evaluate_indices(indices, f"cma_g{generation:04d}")))
            used_solutions.append(vector)
        if len(fitnesses) < 2:
            break
        es.tell(used_solutions, fitnesses)
        generation += 1
        if generation % args.progress_interval == 0 or int(min(fitnesses)) <= best.tokens:
            LOGGER.info(
                "GEN generation=%d evals=%d unique=%d gen_best=%d best=%d elapsed=%.1fs",
                generation,
                eval_count,
                len(cache),
                int(min(fitnesses)),
                int(best.tokens),
                time.monotonic() - start,
            )

    summary = best_payload(best, ordered[:budget], fixed_in, search_candidates, state)
    summary.update(
        {
            "baseline_tokens": baseline_tokens,
            "evals": eval_count,
            "unique_roundings": len(cache),
            "generations": generation,
            "elapsed_seconds": time.monotonic() - start,
            "cma_stop": es.stop(),
            "base_vocab": len(base_vocab),
            "budget": budget,
            "fixed_in": len(fixed_in),
            "search_dim": len(search_candidates),
            "local_budget": local_budget,
        }
    )
    LOGGER.info("SUMMARY %s", json.dumps(compact_payload(summary), ensure_ascii=False, sort_keys=True))
    write_outputs(args, summary, base_vocab, fixed_in, search_candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use CMA-ES to search for better LP vocabulary roundings.")
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--run-dir", help="Training run directory containing lp/training_state.")
    checkpoint.add_argument("--state-dir", help="Direct path to an LP training_state directory.")
    parser.add_argument("--solution-path", help="Optional .npy solution path; defaults to latest_solution.npy in state dir.")
    parser.add_argument("--data-dir", required=True, help="Training/evaluation text directory.")
    parser.add_argument("--output-json", help="Write best/current summary JSON here.")
    parser.add_argument("--output-tokenizer", help="Write the best rounded tokenizer JSON here whenever a new best is found.")
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8, help="Use 0 to match unlimited LP token length.")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--candidate-window", type=int, default=80, help="Optimize this many candidates around the cutoff; 0 means all positive candidates.")
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--popsize", type=int, default=128)
    parser.add_argument("--max-evals", type=int, default=10000)
    parser.add_argument("--time-limit-seconds", type=float, default=0.0, help="Optional wall-clock cap; 0 means no cap.")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--lower-bound", type=float, default=-0.25)
    parser.add_argument("--upper-bound", type=float, default=1.25)
    parser.add_argument("--progress-interval", type=int, default=1)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def resolve_state_dir(args: argparse.Namespace) -> Path:
    if args.state_dir:
        return Path(args.state_dir).expanduser()
    return Path(args.run_dir).expanduser() / "lp" / "training_state"


def resolve_solution_path(args: argparse.Namespace, state_dir: Path) -> Path:
    if args.solution_path:
        return Path(args.solution_path).expanduser()
    return state_dir / "latest_solution.npy"


def load_state(state_dir: Path) -> dict:
    state_path = state_dir / "state.json"
    if not state_path.exists():
        LOGGER.warning("No checkpoint state.json found at %s; continuing with solution only", state_path)
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def build_base_vocab() -> tuple[list[str], set[str]]:
    base_vocab = []
    excluded = set()
    for token in [*DEFAULT_SPECIAL_TOKENS, *byte_level_alphabet()]:
        if token in excluded:
            continue
        base_vocab.append(token)
        excluded.add(token)
    return base_vocab, excluded


def split_search_window(ordered, budget: int, candidate_window: int):
    if candidate_window <= 0:
        return [], ordered, budget
    low = max(0, budget - candidate_window // 2)
    high = min(len(ordered), budget + candidate_window // 2)
    fixed_in = ordered[:low]
    search_candidates = ordered[low:high]
    local_budget = budget - len(fixed_in)
    if local_budget <= 0 or local_budget > len(search_candidates):
        raise ValueError(f"Invalid local budget {local_budget} for search window {len(search_candidates)}")
    return fixed_in, search_candidates, local_budget


def select_from_vector(search_candidates, local_budget: int, vector) -> tuple[int, ...]:
    order = sorted(
        range(len(search_candidates)),
        key=lambda idx: (
            float(vector[idx]),
            search_candidates[idx].lp_value,
            search_candidates[idx].token_instance_count,
            len(search_candidates[idx].token),
        ),
        reverse=True,
    )
    return tuple(sorted(order[:local_budget]))


class SearchBest:
    def __init__(self, *, tokens: float, eval: int, indices: tuple[int, ...], source: str):
        self.tokens = tokens
        self.eval = eval
        self.indices = indices
        self.source = source


def best_payload(best: SearchBest, deterministic, fixed_in, search_candidates, state: dict) -> dict:
    selected = [*fixed_in, *(search_candidates[idx] for idx in best.indices)]
    selected_tokens = {token.token for token in selected}
    deterministic_tokens = {token.token for token in deterministic}
    lp_lower_bound = (state.get("final_metadata") or {}).get("token_count_lower_bound")
    gap = None if lp_lower_bound is None or math.isinf(best.tokens) else float(best.tokens) - float(lp_lower_bound)
    return {
        "tokens": None if math.isinf(best.tokens) else int(best.tokens),
        "eval": int(best.eval),
        "source": best.source,
        "selected_min_lp": min((token.lp_value for token in selected), default=None),
        "selected_avg_lp": (
            sum(token.lp_value for token in selected) / len(selected)
            if selected
            else None
        ),
        "changed_from_deterministic": len(selected_tokens - deterministic_tokens),
        "selected_tokens": [token.token for token in selected],
        "added_tokens": sorted(selected_tokens - deterministic_tokens),
        "removed_tokens": sorted(deterministic_tokens - selected_tokens),
        "checkpoint_next_iteration": state.get("next_iteration"),
        "active_cuts": len(state.get("existing_cut_keys", [])) if state else None,
        "lp_lower_bound": lp_lower_bound,
        "gap_to_lower_bound": gap,
    }


def compact_payload(payload: dict) -> dict:
    compact = dict(payload)
    selected_tokens = compact.pop("selected_tokens", None)
    if selected_tokens is not None:
        compact["selected_token_count"] = len(selected_tokens)
    for key in ("added_tokens", "removed_tokens"):
        values = compact.get(key)
        if isinstance(values, list) and len(values) > 12:
            compact[key] = values[:12]
            compact[f"{key}_truncated"] = len(values)
    return compact


def write_outputs(args: argparse.Namespace, payload: dict, base_vocab: list[str], fixed_in, search_candidates) -> None:
    if args.output_json:
        path = Path(args.output_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_tokenizer and payload.get("tokens") is not None:
        vocab = [*base_vocab, *(payload.get("selected_tokens") or [])]
        path = Path(args.output_tokenizer).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        LpDpTokenizer(vocab, pretokenizer_mode=args.pretokenizer, unk_token=DEFAULT_UNK_TOKEN).save(path)


if __name__ == "__main__":
    main()
