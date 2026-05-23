from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE

from tokenisation_lp.pretokenization import (
    DEFAULT_SPECIAL_TOKENS,
    DEFAULT_UNK_TOKEN,
    build_pretokenizer,
    byte_level_alphabet,
    pretokenize_text,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BpeTrainingResult:
    tokenizer: Tokenizer
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    output_path: Path | None = None


def train_bpe_tokenizer(
    texts: list[str],
    vocab_size: int,
    *,
    pretokenizer_mode: str = "bytelevel",
    special_tokens: list[str] | None = None,
    unk_token: str = DEFAULT_UNK_TOKEN,
    output_dir: str | Path | None = None,
) -> BpeTrainingResult:
    special_tokens = list(special_tokens or DEFAULT_SPECIAL_TOKENS)
    alphabet = byte_level_alphabet()
    if vocab_size < len(special_tokens) + len(alphabet):
        raise ValueError(
            f"vocab_size={vocab_size} is too small for {len(special_tokens)} special "
            f"tokens plus the {len(alphabet)} byte-level alphabet."
        )

    pretokenizer, decoder = build_pretokenizer(pretokenizer_mode)
    word_counts = Counter()
    for text in texts:
        word_counts.update(pretokenize_text(text, pretokenizer))

    LOGGER.info("BPE training corpus has %d unique pretokenized strings", len(word_counts))
    vocab_tokens = dedupe_preserve_order([*special_tokens, *alphabet])
    merges = learn_greedy_merges(word_counts, vocab_tokens, target_vocab_size=vocab_size)
    vocab_tokens.extend("".join(pair) for pair in merges)
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}

    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token=unk_token))
    tokenizer.pre_tokenizer = pretokenizer
    tokenizer.decoder = decoder

    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir) / "bpe_tokenizer.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path))
        LOGGER.info("Saved BPE tokenizer to %s", output_path)

    return BpeTrainingResult(
        tokenizer=tokenizer,
        vocab=vocab,
        merges=merges,
        output_path=output_path,
    )


def learn_greedy_merges(
    word_counts: Counter[str],
    initial_vocab: list[str],
    *,
    target_vocab_size: int,
) -> list[tuple[str, str]]:
    """Learn BPE merges by repeatedly taking the most frequent adjacent pair."""

    tokenized_words: dict[tuple[str, ...], int] = {
        tuple(word): count for word, count in word_counts.items() if word
    }
    merges: list[tuple[str, str]] = []
    vocab = set(initial_vocab)

    while len(vocab) < target_vocab_size:
        pair_counts = count_adjacent_pairs(tokenized_words)
        if not pair_counts:
            break

        ranked_pairs = sorted(
            pair_counts.items(),
            key=lambda item: (item[1], len(item[0][0]) + len(item[0][1]), item[0]),
            reverse=True,
        )
        best_pair = None
        best_count = 0
        merged_token = ""
        for pair, count in ranked_pairs:
            candidate = "".join(pair)
            if candidate not in vocab:
                best_pair = pair
                best_count = count
                merged_token = candidate
                break

        if best_pair is None:
            break

        tokenized_words = merge_pair_in_words(tokenized_words, best_pair, merged_token)
        merges.append(best_pair)
        vocab.add(merged_token)

        if len(merges) == 1 or len(merges) % 100 == 0:
            LOGGER.info(
                "BPE merge %d/%d: %r + %r -> %r (count=%d)",
                len(merges),
                target_vocab_size - len(initial_vocab),
                best_pair[0],
                best_pair[1],
                merged_token,
                best_count,
            )

    LOGGER.info("Learned %d greedy BPE merges", len(merges))
    return merges


def count_adjacent_pairs(tokenized_words: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for pieces, freq in tokenized_words.items():
        for left, right in zip(pieces, pieces[1:]):
            counts[(left, right)] += freq
    return counts


def merge_pair_in_words(
    tokenized_words: dict[tuple[str, ...], int],
    pair: tuple[str, str],
    merged_token: str,
) -> dict[tuple[str, ...], int]:
    merged_words: Counter[tuple[str, ...]] = Counter()
    left, right = pair

    for pieces, freq in tokenized_words.items():
        new_pieces = []
        i = 0
        while i < len(pieces):
            if i + 1 < len(pieces) and pieces[i] == left and pieces[i + 1] == right:
                new_pieces.append(merged_token)
                i += 2
            else:
                new_pieces.append(pieces[i])
                i += 1
        merged_words[tuple(new_pieces)] += freq

    return dict(merged_words)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
