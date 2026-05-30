from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tokenisation_lp.cma_rounding_search import build_base_vocab, compact_payload
from tokenisation_lp.corpus import load_texts
from tokenisation_lp.dp_tokenizer import LpDpTokenizer
from tokenisation_lp.lp_training import (
    build_standard_form,
    candidates_from_solution,
    count_pretokenized_strings,
    prepare_lp_data,
)
from tokenisation_lp.pretokenization import DEFAULT_UNK_TOKEN, build_pretokenizer


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Move:
    remove: tuple[int, ...]
    add: tuple[int, ...]
    delta: int


@dataclass
class BestState:
    tokens: int
    selected: frozenset[int]
    iteration: int
    source: str


class IncrementalTokenCounter:
    def __init__(self, words: list[str], freqs: list[int], candidate_tokens: list[str]):
        self.words = words
        self.freqs = np.asarray(freqs, dtype=np.int64)
        self.candidate_tokens = candidate_tokens
        self.word_edges, self.token_words = build_word_token_incidence(words, candidate_tokens)
        self.word_costs = np.zeros(len(words), dtype=np.int16)
        self.selected = np.zeros(len(candidate_tokens), dtype=bool)
        self.total_tokens = 0

    def initialize(self, selected_indices: set[int]) -> int:
        self.selected[:] = False
        if selected_indices:
            self.selected[list(selected_indices)] = True
        total = 0
        for word_idx in range(len(self.words)):
            cost = self.word_cost(word_idx, self.selected)
            self.word_costs[word_idx] = cost
            total += int(cost) * int(self.freqs[word_idx])
        self.total_tokens = int(total)
        return self.total_tokens

    def evaluate_move(self, remove: tuple[int, ...], add: tuple[int, ...]) -> int:
        affected = self.affected_words(remove, add)
        if not affected:
            return 0
        moved = self.selected.copy()
        if remove:
            moved[list(remove)] = False
        if add:
            moved[list(add)] = True
        delta = 0
        for word_idx in affected:
            new_cost = self.word_cost(word_idx, moved)
            delta += (int(new_cost) - int(self.word_costs[word_idx])) * int(self.freqs[word_idx])
        return int(delta)

    def apply_move(self, remove: tuple[int, ...], add: tuple[int, ...]) -> int:
        affected = self.affected_words(remove, add)
        if remove:
            self.selected[list(remove)] = False
        if add:
            self.selected[list(add)] = True
        delta = 0
        for word_idx in affected:
            new_cost = self.word_cost(word_idx, self.selected)
            delta += (int(new_cost) - int(self.word_costs[word_idx])) * int(self.freqs[word_idx])
            self.word_costs[word_idx] = new_cost
        self.total_tokens += int(delta)
        return int(delta)

    def affected_words(self, remove: tuple[int, ...], add: tuple[int, ...]) -> list[int]:
        affected: set[int] = set()
        for token_idx in (*remove, *add):
            affected.update(self.token_words[token_idx])
        return list(affected)

    def word_cost(self, word_idx: int, selected) -> int:
        word = self.words[word_idx]
        n = len(word)
        costs = [0] * (n + 1)
        for start in range(n - 1, -1, -1):
            best = 1 + costs[start + 1]
            for end, token_idx in self.word_edges[word_idx][start]:
                if selected[token_idx]:
                    candidate = 1 + costs[end]
                    if candidate < best:
                        best = candidate
            costs[start] = best
        return int(costs[0])


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    start_time = time.monotonic()
    rng = random.Random(args.seed)

    state_dir = resolve_state_dir(args)
    state = load_state(state_dir)
    solution = np.load(resolve_solution_path(args, state_dir), allow_pickle=False)

    LOGGER.info("Loading corpus and rebuilding candidate list")
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
    if args.max_candidates > 0:
        ordered = ordered[: args.max_candidates]
    if len(ordered) < budget:
        raise ValueError(f"Only {len(ordered)} candidates available for budget {budget}")

    selected = initial_selection(args, ordered, budget, excluded)
    LOGGER.info(
        "Building incremental evaluator: words=%d candidates=%d budget=%d selected=%d",
        len(words),
        len(ordered),
        budget,
        len(selected),
    )
    counter = IncrementalTokenCounter(words, freqs, [token.token for token in ordered])
    current_tokens = counter.initialize(selected)
    selected = set(np.flatnonzero(counter.selected))
    best = BestState(
        tokens=current_tokens,
        selected=frozenset(selected),
        iteration=0,
        source="initial",
    )
    write_outputs(args, best, ordered, base_vocab, state)
    LOGGER.info("INITIAL %s", json.dumps(compact_payload(best_payload(best, ordered, state)), ensure_ascii=False, sort_keys=True))

    deadline = start_time + args.time_limit_seconds if args.time_limit_seconds > 0 else None
    local_minima = 0
    accepted_moves = 0
    evaluated_moves = 0

    for iteration in range(1, args.max_iterations + 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        move, evals = find_best_move(args, counter, rng, deadline, iteration)
        evaluated_moves += evals
        if move is not None and move.delta < -args.min_improvement:
            applied_delta = counter.apply_move(move.remove, move.add)
            if applied_delta != move.delta:
                raise RuntimeError(f"Move delta changed from {move.delta} to {applied_delta}")
            selected = set(np.flatnonzero(counter.selected))
            accepted_moves += 1
            LOGGER.info(
                "MOVE iteration=%d k=%d delta=%d tokens=%d remove=%s add=%s evals=%d elapsed=%.1fs",
                iteration,
                len(move.remove),
                move.delta,
                counter.total_tokens,
                token_names(ordered, move.remove),
                token_names(ordered, move.add),
                evals,
                time.monotonic() - start_time,
            )
            if counter.total_tokens < best.tokens:
                best = BestState(
                    tokens=counter.total_tokens,
                    selected=frozenset(selected),
                    iteration=iteration,
                    source=f"move_{accepted_moves}",
                )
                LOGGER.info("NEW_BEST %s", json.dumps(compact_payload(best_payload(best, ordered, state)), ensure_ascii=False, sort_keys=True))
                write_outputs(args, best, ordered, base_vocab, state)
            continue

        local_minima += 1
        LOGGER.info(
            "LOCAL_MIN iteration=%d tokens=%d best=%d local_minima=%d evals=%d elapsed=%.1fs",
            iteration,
            counter.total_tokens,
            best.tokens,
            local_minima,
            evals,
            time.monotonic() - start_time,
        )
        if local_minima > args.noise_restarts:
            break
        noise_move = choose_noise_move(args, counter, rng)
        if noise_move is None:
            break
        delta = counter.evaluate_move(noise_move.remove, noise_move.add)
        if args.noise_max_worsen >= 0 and delta > args.noise_max_worsen:
            LOGGER.info(
                "NOISE_SKIPPED iteration=%d k=%d delta=%d max_worsen=%d",
                iteration,
                len(noise_move.remove),
                delta,
                args.noise_max_worsen,
            )
            counter.initialize(set(best.selected))
        else:
            counter.apply_move(noise_move.remove, noise_move.add)
            LOGGER.info(
                "NOISE iteration=%d k=%d delta=%d tokens=%d remove=%s add=%s",
                iteration,
                len(noise_move.remove),
                delta,
                counter.total_tokens,
                token_names(ordered, noise_move.remove),
                token_names(ordered, noise_move.add),
            )

    summary = best_payload(best, ordered, state)
    summary.update(
        {
            "initial_tokens": current_tokens,
            "final_current_tokens": counter.total_tokens,
            "accepted_moves": accepted_moves,
            "evaluated_moves": evaluated_moves,
            "local_minima": local_minima,
            "elapsed_seconds": time.monotonic() - start_time,
            "budget": budget,
            "candidate_count": len(ordered),
        }
    )
    LOGGER.info("SUMMARY %s", json.dumps(compact_payload(summary), ensure_ascii=False, sort_keys=True))
    write_outputs(args, best, ordered, base_vocab, state, summary=summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy 1/2/3-opt local search for LP rounded vocabularies.")
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--run-dir", help="Training run directory containing lp/training_state.")
    checkpoint.add_argument("--state-dir", help="Direct path to an LP training_state directory.")
    parser.add_argument("--solution-path", help="Optional .npy solution path; defaults to latest_solution.npy in state dir.")
    parser.add_argument("--data-dir", required=True, help="Training/evaluation text directory.")
    parser.add_argument("--output-json", help="Write best/current summary JSON here.")
    parser.add_argument("--output-tokenizer", help="Write the best rounded tokenizer JSON here whenever a new best is found.")
    parser.add_argument("--seed-json", help="Optional CMA/3-opt summary JSON with selected_tokens.")
    parser.add_argument("--seed-tokenizer", help="Optional LP DP tokenizer JSON to use as the initial vocabulary.")
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8, help="Use 0 to match unlimited LP token length.")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=0, help="Limit search to top LP candidates; 0 means all positive candidates.")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--max-swap-size", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--remove-pool", type=int, default=96)
    parser.add_argument("--add-pool", type=int, default=96)
    parser.add_argument("--exhaustive-1-pool", type=int, default=96)
    parser.add_argument("--exhaustive-2-pool", type=int, default=18)
    parser.add_argument("--exhaustive-3-pool", type=int, default=10)
    parser.add_argument("--random-proposals", type=int, default=20000)
    parser.add_argument(
        "--move-time-limit-seconds",
        type=float,
        default=0.0,
        help="Optional wall-clock cap for one best-move search; 0 means no per-iteration cap.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=30.0,
        help="Log best-move search progress at this interval; 0 disables progress logs.",
    )
    parser.add_argument("--min-improvement", type=int, default=0)
    parser.add_argument("--noise-restarts", type=int, default=5)
    parser.add_argument("--noise-swaps", type=int, default=3)
    parser.add_argument("--noise-max-worsen", type=int, default=-1, help="Maximum accepted noise worsening; negative means unlimited.")
    parser.add_argument("--time-limit-seconds", type=float, default=0.0, help="Optional wall-clock cap; 0 means no cap.")
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def find_best_move(
    args,
    counter: IncrementalTokenCounter,
    rng: random.Random,
    deadline: float | None,
    iteration: int,
) -> tuple[Move | None, int]:
    selected = tuple(int(idx) for idx in np.flatnonzero(counter.selected))
    unselected = tuple(int(idx) for idx in np.flatnonzero(~counter.selected))
    remove_scores = rank_removals(counter, selected)
    add_scores = rank_additions(counter, unselected)
    remove_pool = tuple(idx for _delta, idx in remove_scores[: args.remove_pool])
    add_pool = tuple(idx for _delta, idx in add_scores[: args.add_pool])
    best: Move | None = None
    evaluated = 0
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    search_start = time.monotonic()
    search_deadline = deadline
    if args.move_time_limit_seconds > 0:
        move_deadline = search_start + args.move_time_limit_seconds
        search_deadline = move_deadline if search_deadline is None else min(search_deadline, move_deadline)
    next_progress = (
        search_start + args.progress_interval_seconds
        if args.progress_interval_seconds > 0
        else math.inf
    )

    LOGGER.info(
        "SEARCH_START iteration=%d tokens=%d remove_pool=%d add_pool=%d max_swap=%d",
        iteration,
        counter.total_tokens,
        len(remove_pool),
        len(add_pool),
        args.max_swap_size,
    )

    def should_stop() -> bool:
        return search_deadline is not None and time.monotonic() >= search_deadline

    def maybe_log_progress(k: int, phase: str) -> None:
        nonlocal next_progress
        now = time.monotonic()
        if now < next_progress:
            return
        elapsed = now - search_start
        rate = evaluated / elapsed if elapsed > 0 else 0.0
        LOGGER.info(
            "SEARCH_PROGRESS iteration=%d k=%d phase=%s evals=%d rate=%.1f/s best_delta=%s elapsed=%.1fs",
            iteration,
            k,
            phase,
            evaluated,
            rate,
            None if best is None else best.delta,
            elapsed,
        )
        while next_progress <= now:
            next_progress += args.progress_interval_seconds

    def consider(remove, add) -> None:
        nonlocal best, evaluated
        if should_stop():
            return
        remove = tuple(sorted(remove))
        add = tuple(sorted(add))
        if len(remove) != len(add) or not remove or not add:
            return
        key = (remove, add)
        if key in seen:
            return
        seen.add(key)
        delta = counter.evaluate_move(remove, add)
        evaluated += 1
        if best is None or delta < best.delta:
            best = Move(remove=remove, add=add, delta=delta)

    for k in range(1, args.max_swap_size + 1):
        exhaustive_pool = {
            1: args.exhaustive_1_pool,
            2: args.exhaustive_2_pool,
            3: args.exhaustive_3_pool,
        }[k]
        if exhaustive_pool > 0:
            for remove in itertools.combinations(remove_pool[:exhaustive_pool], k):
                for add in itertools.combinations(add_pool[:exhaustive_pool], k):
                    consider(remove, add)
                    maybe_log_progress(k, "exhaustive")
                    if should_stop():
                        return best, evaluated
        proposals = max(0, int(args.random_proposals))
        for _ in range(proposals):
            if len(remove_pool) < k or len(add_pool) < k:
                break
            consider(rng.sample(remove_pool, k), rng.sample(add_pool, k))
            maybe_log_progress(k, "random")
            if should_stop():
                return best, evaluated
    return best, evaluated


def rank_removals(counter: IncrementalTokenCounter, selected: tuple[int, ...]) -> list[tuple[int, int]]:
    rows = []
    for token_idx in selected:
        delta = counter.evaluate_move((token_idx,), ())
        rows.append((delta, token_idx))
    rows.sort(key=lambda item: item[0])
    return rows


def rank_additions(counter: IncrementalTokenCounter, unselected: tuple[int, ...]) -> list[tuple[int, int]]:
    rows = []
    for token_idx in unselected:
        delta = counter.evaluate_move((), (token_idx,))
        rows.append((delta, token_idx))
    rows.sort(key=lambda item: item[0])
    return rows


def choose_noise_move(args, counter: IncrementalTokenCounter, rng: random.Random) -> Move | None:
    selected = tuple(int(idx) for idx in np.flatnonzero(counter.selected))
    unselected = tuple(int(idx) for idx in np.flatnonzero(~counter.selected))
    k = min(max(1, int(args.noise_swaps)), len(selected), len(unselected))
    if k <= 0:
        return None
    return Move(
        remove=tuple(sorted(rng.sample(selected, k))),
        add=tuple(sorted(rng.sample(unselected, k))),
        delta=0,
    )


def build_word_token_incidence(words: list[str], candidate_tokens: list[str]):
    word_edges = [[[] for _ in word] for word in words]
    token_word_sets = [set() for _ in candidate_tokens]
    for token_idx, token in enumerate(candidate_tokens):
        if not token:
            continue
        token_len = len(token)
        for word_idx, word in enumerate(words):
            start = word.find(token)
            while start >= 0:
                word_edges[word_idx][start].append((start + token_len, token_idx))
                token_word_sets[token_idx].add(word_idx)
                start = word.find(token, start + 1)
    return word_edges, [tuple(sorted(word_set)) for word_set in token_word_sets]


def initial_selection(args, ordered, budget: int, excluded: set[str]) -> set[int]:
    token_to_idx = {token.token: idx for idx, token in enumerate(ordered)}
    selected_tokens: list[str] = []
    if args.seed_json:
        payload = json.loads(Path(args.seed_json).expanduser().read_text(encoding="utf-8"))
        selected_tokens.extend(str(token) for token in payload.get("selected_tokens") or [])
    if args.seed_tokenizer:
        payload = json.loads(Path(args.seed_tokenizer).expanduser().read_text(encoding="utf-8"))
        selected_tokens.extend(str(token) for token in payload.get("vocab") or [])

    selected = []
    seen = set()
    ignored = 0
    for token in selected_tokens:
        if token in excluded or token in seen:
            continue
        idx = token_to_idx.get(token)
        if idx is None:
            ignored += 1
            continue
        selected.append(idx)
        seen.add(token)
    if ignored:
        LOGGER.warning("Ignored %d seed tokens that are not searchable LP candidates", ignored)
    if len(selected) > budget:
        selected = sorted(selected, key=lambda idx: ordered[idx].lp_value, reverse=True)[:budget]
    for idx in range(len(ordered)):
        if len(selected) >= budget:
            break
        if idx not in selected:
            selected.append(idx)
    return set(selected)


def best_payload(best: BestState, ordered, state: dict) -> dict:
    selected = [ordered[idx] for idx in sorted(best.selected)]
    selected_tokens = [token.token for token in selected]
    lp_lower_bound = (state.get("final_metadata") or {}).get("token_count_lower_bound")
    gap = None if lp_lower_bound is None else float(best.tokens) - float(lp_lower_bound)
    return {
        "tokens": int(best.tokens),
        "iteration": int(best.iteration),
        "source": best.source,
        "selected_tokens": selected_tokens,
        "selected_token_count": len(selected_tokens),
        "selected_min_lp": min((token.lp_value for token in selected), default=None),
        "selected_avg_lp": (
            sum(token.lp_value for token in selected) / len(selected)
            if selected
            else None
        ),
        "checkpoint_next_iteration": state.get("next_iteration"),
        "active_cuts": len(state.get("existing_cut_keys", [])) if state else None,
        "lp_lower_bound": lp_lower_bound,
        "gap_to_lower_bound": gap,
    }


def write_outputs(args, best: BestState, ordered, base_vocab: list[str], state: dict, *, summary: dict | None = None) -> None:
    payload = summary if summary is not None else best_payload(best, ordered, state)
    if args.output_json:
        path = Path(args.output_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_tokenizer:
        selected_tokens = [ordered[idx].token for idx in sorted(best.selected)]
        path = Path(args.output_tokenizer).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        LpDpTokenizer(
            [*base_vocab, *selected_tokens],
            pretokenizer_mode=args.pretokenizer,
            unk_token=DEFAULT_UNK_TOKEN,
        ).save(path)


def token_names(ordered, indices: tuple[int, ...]) -> list[str]:
    return [ordered[idx].token for idx in indices]


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


if __name__ == "__main__":
    main()
