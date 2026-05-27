from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tokenisation_lp.corpus import load_texts
from tokenisation_lp.lp_training import build_standard_form, count_pretokenized_strings, prepare_lp_data
from tokenisation_lp.pretokenization import build_pretokenizer


LOGGER = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    cuts_path = Path(args.cuts_jsonl).expanduser()

    LOGGER.info("Loading cuts from %s", cuts_path)
    records = load_jsonl(cuts_path)
    LOGGER.info("Loaded cuts=%d", len(records))

    LOGGER.info("Rebuilding LP metadata")
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

    analyses = [analyze_cut(record, lp, words, tokens) for record in records]
    summary = summarize(analyses, args)

    summary_path = output_dir / "summary.json"
    examples_path = output_dir / "template_examples.jsonl"
    log_path = output_dir / "template_analysis.log"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_examples(examples_path, analyses, args)
    log_lines = format_log(summary, analyses, args)
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    for line in log_lines:
        LOGGER.info(line)
    LOGGER.info("Wrote summary=%s examples=%s log=%s", summary_path, examples_path, log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify brute-force reduced pair cuts into reusable templates.")
    parser.add_argument("--cuts-jsonl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pretokenizer", default="nanochat", choices=("bytelevel", "split_bytelevel", "apertus", "nanochat"))
    parser.add_argument("--min-token-count", type=int, default=5)
    parser.add_argument("--max-token-length", type=int, default=8)
    parser.add_argument("--top-groups", type=int, default=40)
    parser.add_argument("--sample-per-group", type=int, default=5)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_cut(record: dict, lp: dict, words: list[str], tokens: list) -> dict:
    num_f = int(lp["num_nonfree_edges"])
    num_g = int(lp["num_free_edges"])
    t_offset = num_f + num_g
    left_word = int(record["left_word_idx"])
    right_word = int(record["right_word_idx"])
    pair_words = (left_word, right_word)

    edge_terms = []
    token_terms = []
    for col_idx, coeff in record["entries"]:
        col_idx = int(col_idx)
        coeff = float(coeff)
        if col_idx >= t_offset:
            token_idx = col_idx - t_offset
            token_terms.append(
                {
                    "token_idx": int(token_idx),
                    "token": tokens[token_idx].token,
                    "coefficient": coeff,
                }
            )
        elif col_idx < num_f:
            info = lp["nonfree_edge_info"][col_idx]
            edge_terms.append(edge_term(col_idx, coeff, info, "nonfree", words))
        else:
            info = lp["free_edge_info"][col_idx - num_f]
            edge_terms.append(edge_term(col_idx, coeff, info, "free", words))

    scale = coefficient_scale(edge_terms, token_terms, float(record["rhs"]))
    for term in edge_terms:
        term["norm_coefficient"] = normalized_value(term["coefficient"], scale)
    for term in token_terms:
        term["norm_coefficient"] = normalized_value(term["coefficient"], scale)
    rhs_norm = normalized_value(float(record["rhs"]), scale)

    edge_by_word = {word_idx: [] for word_idx in pair_words}
    for term in edge_terms:
        edge_by_word.setdefault(int(term["word_idx"]), []).append(term)
    for terms in edge_by_word.values():
        terms.sort(key=lambda term: (term["start"], term["end"], term["kind"], term["token"], term["norm_coefficient"]))

    shape_key = make_shape_key(pair_words, edge_by_word, token_terms, rhs_norm)
    family = classify_family(edge_terms, token_terms, rhs_norm)
    selected_token_strings = [
        tokens[int(token_idx)].token
        for token_idx in record.get("key", [None, None, None, []])[3]
    ] if len(record.get("key", [])) > 3 else []

    return {
        "rank": int(record.get("rank", 0)),
        "violation": float(record.get("violation", 0.0)),
        "rhs": float(record["rhs"]),
        "rhs_norm": rhs_norm,
        "scale": scale,
        "left_word_idx": left_word,
        "right_word_idx": right_word,
        "left_word": record.get("left_word", words[left_word]),
        "right_word": record.get("right_word", words[right_word]),
        "selected_tokens": selected_token_strings,
        "edge_count_pattern": tuple(len(edge_by_word.get(word_idx, [])) for word_idx in pair_words),
        "edge_sign_pattern": tuple(sign_counts(edge_by_word.get(word_idx, [])) for word_idx in pair_words),
        "edge_kind_pattern": tuple(kind_counts(edge_by_word.get(word_idx, [])) for word_idx in pair_words),
        "token_count": len(token_terms),
        "token_sign_pattern": sign_counts(token_terms),
        "coefficient_pattern": coefficient_pattern(edge_terms, token_terms, rhs_norm),
        "shape_key": shape_key,
        "family": family,
        "row_counts": {
            "dedup": int(record.get("dedup_rows", 0)),
            "original": int(record.get("original_rows", 0)),
            "edge_vars": int(record.get("edge_vars", 0)),
        },
        "edges_by_word": [
            {
                "word_idx": int(word_idx),
                "word": words[word_idx],
                "edges": edge_by_word.get(word_idx, []),
            }
            for word_idx in pair_words
        ],
        "token_terms": sorted(token_terms, key=lambda term: (term["norm_coefficient"], term["token"])),
    }


def edge_term(col_idx: int, coeff: float, info: dict, kind: str, words: list[str]) -> dict:
    return {
        "col_idx": int(col_idx),
        "word_idx": int(info["word_idx"]),
        "word": words[int(info["word_idx"])],
        "start": int(info["start"]),
        "end": int(info["end"]),
        "length": int(info["end"]) - int(info["start"]),
        "token": info["token"],
        "kind": kind,
        "coefficient": coeff,
    }


def coefficient_scale(edge_terms: list[dict], token_terms: list[dict], rhs: float) -> float:
    values = [abs(float(term["coefficient"])) for term in edge_terms + token_terms]
    if abs(rhs) > 1e-12:
        values.append(abs(float(rhs)))
    positive = [value for value in values if value > 1e-10]
    return min(positive) if positive else 1.0


def normalized_value(value: float, scale: float):
    ratio = float(value) / scale
    rounded = round(ratio)
    if abs(ratio - rounded) <= 1e-7:
        return int(rounded)
    return round(ratio, 8)


def make_shape_key(pair_words: tuple[int, int], edge_by_word: dict[int, list[dict]], token_terms: list[dict], rhs_norm) -> str:
    words_part = []
    for word_idx in pair_words:
        rel_edges = []
        for term in edge_by_word.get(word_idx, []):
            rel_edges.append(
                (
                    int(term["start"]),
                    int(term["end"]),
                    term["kind"],
                    sign(term["coefficient"]),
                    term["norm_coefficient"],
                )
            )
        words_part.append(tuple(rel_edges))
    token_part = tuple(sorted((sign(term["coefficient"]), term["norm_coefficient"]) for term in token_terms))
    return repr((tuple(words_part), token_part, rhs_norm))


def classify_family(edge_terms: list[dict], token_terms: list[dict], rhs_norm) -> str:
    if not edge_terms:
        return "no_edges"
    edge_coeffs = {term["norm_coefficient"] for term in edge_terms}
    token_coeffs = {term["norm_coefficient"] for term in token_terms}
    edge_signs = {sign(term["coefficient"]) for term in edge_terms}
    token_signs = {sign(term["coefficient"]) for term in token_terms}
    if edge_signs == {"+"} and token_signs == {"-"} and len(edge_coeffs) == 1 and len(token_coeffs) == 1:
        if next(iter(edge_coeffs)) == -next(iter(token_coeffs)):
            if rhs_norm == next(iter(edge_coeffs)):
                return "uniform_edges_tokens_rhs1"
            return "uniform_edges_tokens"
    if edge_signs == {"+"} and token_signs <= {"-"} and len(token_terms) == 2:
        return "positive_edges_two_negative_tokens"
    if edge_signs == {"+"} and token_signs <= {"-"}:
        return "positive_edges_negative_tokens"
    if "-" in edge_signs:
        return "mixed_edge_signs"
    if "+" in token_signs:
        return "positive_token_coefficients"
    return "other"


def sign_counts(terms: list[dict]) -> tuple[tuple[str, int], ...]:
    counts = Counter(sign(term["coefficient"]) for term in terms)
    return tuple(sorted(counts.items()))


def kind_counts(terms: list[dict]) -> tuple[tuple[str, int], ...]:
    counts = Counter(term.get("kind", "token") for term in terms)
    return tuple(sorted(counts.items()))


def sign(value: float) -> str:
    if value > 0:
        return "+"
    if value < 0:
        return "-"
    return "0"


def coefficient_pattern(edge_terms: list[dict], token_terms: list[dict], rhs_norm) -> dict:
    return {
        "edge": dict(Counter(term["norm_coefficient"] for term in edge_terms)),
        "token": dict(Counter(term["norm_coefficient"] for term in token_terms)),
        "rhs": rhs_norm,
    }


def summarize(analyses: list[dict], args: argparse.Namespace) -> dict:
    families = Counter(item["family"] for item in analyses)
    edge_patterns = Counter(item["edge_count_pattern"] for item in analyses)
    sign_patterns = Counter((item["edge_sign_pattern"], item["token_sign_pattern"]) for item in analyses)
    coefficient_patterns = Counter(json.dumps(item["coefficient_pattern"], sort_keys=True) for item in analyses)
    shapes = Counter(item["shape_key"] for item in analyses)
    violations = [item["violation"] for item in analyses]
    rows = [item["row_counts"]["dedup"] for item in analyses]
    return {
        "cuts": len(analyses),
        "violation": quantiles(violations),
        "dedup_rows": quantiles(rows),
        "families": top_counter(families, args.top_groups),
        "edge_count_patterns": top_counter(edge_patterns, args.top_groups),
        "sign_patterns": top_counter(sign_patterns, args.top_groups),
        "coefficient_patterns": top_counter(coefficient_patterns, args.top_groups),
        "shape_groups": {
            "distinct": len(shapes),
            "top": top_counter(shapes, args.top_groups),
        },
    }


def write_examples(path: Path, analyses: list[dict], args: argparse.Namespace) -> None:
    by_shape = defaultdict(list)
    for item in sorted(analyses, key=lambda analysis: analysis["violation"], reverse=True):
        by_shape[item["shape_key"]].append(item)
    shapes = sorted(by_shape.items(), key=lambda kv: (len(kv[1]), kv[1][0]["violation"]), reverse=True)
    with path.open("w", encoding="utf-8") as handle:
        for shape_rank, (shape_key, items) in enumerate(shapes[: args.top_groups], start=1):
            payload = {
                "shape_rank": shape_rank,
                "count": len(items),
                "shape_key": shape_key,
                "family": items[0]["family"],
                "edge_count_pattern": items[0]["edge_count_pattern"],
                "coefficient_pattern": items[0]["coefficient_pattern"],
                "examples": [strip_large_fields(item) for item in items[: args.sample_per_group]],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def strip_large_fields(item: dict) -> dict:
    return {
        "rank": item["rank"],
        "violation": item["violation"],
        "left_word": item["left_word"],
        "right_word": item["right_word"],
        "selected_tokens": item["selected_tokens"],
        "rhs_norm": item["rhs_norm"],
        "scale": item["scale"],
        "row_counts": item["row_counts"],
        "edges_by_word": item["edges_by_word"],
        "token_terms": item["token_terms"],
    }


def format_log(summary: dict, analyses: list[dict], args: argparse.Namespace) -> list[str]:
    lines = [
        f"pair cut template diagnostics: cuts={summary['cuts']}",
        f"violation={summary['violation']}",
        f"dedup_rows={summary['dedup_rows']}",
        "families:",
    ]
    lines.extend(f"  {name}: {count}" for name, count in summary["families"])
    lines.append("edge count patterns:")
    lines.extend(f"  {pattern}: {count}" for pattern, count in summary["edge_count_patterns"][:15])
    lines.append("top shape examples:")
    by_shape = defaultdict(list)
    for item in sorted(analyses, key=lambda analysis: analysis["violation"], reverse=True):
        by_shape[item["shape_key"]].append(item)
    shapes = sorted(by_shape.values(), key=lambda items: (len(items), items[0]["violation"]), reverse=True)
    for shape_rank, items in enumerate(shapes[: min(10, args.top_groups)], start=1):
        first = items[0]
        lines.append(
            f"  #{shape_rank} count={len(items)} family={first['family']} "
            f"edges={first['edge_count_pattern']} coeffs={first['coefficient_pattern']}"
        )
        for example in items[: min(3, args.sample_per_group)]:
            lines.append(
                f"    v={example['violation']:.8g} words=({example['left_word']!r}, {example['right_word']!r}) "
                f"tokens={example['selected_tokens']}"
            )
    return lines


def top_counter(counter: Counter, limit: int) -> list:
    return [[serializable(key), int(value)] for key, value in counter.most_common(limit)]


def quantiles(values: list[float]) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.5)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.9)),
        "max": float(np.max(arr)),
    }


def serializable(value):
    if isinstance(value, tuple):
        return [serializable(item) for item in value]
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
