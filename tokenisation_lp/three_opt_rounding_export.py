from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from tokenisation_lp.cma_rounding_search import build_base_vocab
from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import (
    build_standard_form,
    count_pretokenized_strings,
    prepare_lp_data,
)
from tokenisation_lp.pretokenization import DEFAULT_UNK_TOKEN, build_pretokenizer


LOGGER = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    state_dir = resolve_state_dir(args)
    state = load_state(state_dir)
    solution = np.load(resolve_solution_path(args, state_dir), allow_pickle=False)

    LOGGER.info("Loading corpus and rebuilding LP token universe")
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

    t_values = solution[lp["num_nonfree_edges"] + lp["num_free_edges"] :]
    candidates = []
    skipped_zero = 0
    for idx, token in enumerate(tokens):
        if token.token in excluded:
            continue
        lp_value = float(t_values[idx])
        if args.positive_only and lp_value <= args.positive_tolerance:
            skipped_zero += 1
            continue
        candidates.append(
            {
                "token": token.token,
                "lp_value": lp_value,
                "instance_count": int(token.token_instance_count),
                "index": int(idx),
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item["lp_value"]),
            int(item["instance_count"]),
            len(str(item["token"])),
            str(item["token"]),
        ),
        reverse=True,
    )
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    if len(candidates) < budget:
        raise ValueError(f"Only {len(candidates)} candidates available for budget {budget}")

    payload = {
        "words": [
            {"text": word, "freq": int(freq)}
            for word, freq in zip(words, freqs)
        ],
        "candidates": candidates,
        "base_vocab": base_vocab,
        "budget": int(budget),
        "pretokenizer_mode": args.pretokenizer,
        "unk_token": DEFAULT_UNK_TOKEN,
        "state": {
            "checkpoint_next_iteration": state.get("next_iteration"),
            "active_cuts": len(state.get("existing_cut_keys", [])) if state else None,
            "lp_lower_bound": (state.get("final_metadata") or {}).get("token_count_lower_bound"),
        },
        "stats": {
            "word_count": len(words),
            "candidate_count": len(candidates),
            "positive_candidates": int(np.count_nonzero(t_values > args.positive_tolerance)),
            "skipped_zero_candidates": int(skipped_zero),
            "min_token_count": args.min_token_count,
            "max_token_length": args.max_token_length,
        },
    }

    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Wrote %s: words=%d candidates=%d positive=%d skipped_zero=%d budget=%d",
        output_path,
        len(words),
        len(candidates),
        payload["stats"]["positive_candidates"],
        skipped_zero,
        budget,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a 3-opt rounding search instance for the Go optimizer.")
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--run-dir", help="Training run directory containing lp/training_state.")
    checkpoint.add_argument("--state-dir", help="Direct path to an LP training_state directory.")
    parser.add_argument("--solution-path", help="Optional .npy solution path; defaults to latest_solution.npy in state dir.")
    parser.add_argument("--data-dir", required=True, help="Training/evaluation text directory.")
    parser.add_argument("--output-json", required=True, help="Write exported search instance JSON here.")
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8, help="Use 0 to match unlimited LP token length.")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=0, help="Limit exported candidates after ranking; 0 means all.")
    parser.add_argument("--positive-only", action="store_true", help="Keep the old behavior and export only positive-LP candidates.")
    parser.add_argument("--positive-tolerance", type=float, default=1e-9)
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


if __name__ == "__main__":
    main()
