from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenisation_lp.pretokenization import DEFAULT_UNK_TOKEN, build_pretokenizer, pretokenize_text


@dataclass(frozen=True)
class Tokenization:
    tokens: list[str]
    ids: list[int]


@dataclass
class _TrieNode:
    children: dict[str, "_TrieNode"]
    token_id: int | None = None
    token: str | None = None


class LpDpTokenizer:
    """Optimal tokenizer for a fixed LP vocabulary.

    Each pretokenized string is segmented with dynamic programming. The default
    objective is minimum token count, with token-id tie breaking for
    deterministic output. This is the exact shortest path over all vocabulary
    tokens that match the pretokenized piece.
    """

    def __init__(
        self,
        vocab: list[str],
        *,
        pretokenizer_mode: str = "bytelevel",
        unk_token: str = DEFAULT_UNK_TOKEN,
    ):
        if len(vocab) != len(set(vocab)):
            raise ValueError("LP DP tokenizer vocab contains duplicate tokens")
        if unk_token not in vocab:
            raise ValueError(f"UNK token {unk_token!r} must be present in the vocab")

        self.vocab = list(vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.pretokenizer_mode = pretokenizer_mode
        self.unk_token = unk_token
        self.unk_id = self.token_to_id[unk_token]
        self._pretokenizer, self._decoder = build_pretokenizer(pretokenizer_mode)
        self._trie = build_trie(self.vocab)

    def encode(self, text: str) -> Tokenization:
        tokens: list[str] = []
        ids: list[int] = []
        for piece in pretokenize_text(text, self._pretokenizer):
            piece_tokenization = self.encode_piece(piece)
            tokens.extend(piece_tokenization.tokens)
            ids.extend(piece_tokenization.ids)
        return Tokenization(tokens=tokens, ids=ids)

    def encode_piece(self, piece: str) -> Tokenization:
        n = len(piece)
        best_cost = [float("inf")] * (n + 1)
        best_next: list[tuple[int, str, int] | None] = [None] * (n + 1)
        best_cost[n] = 0

        for start in range(n - 1, -1, -1):
            node = self._trie
            for end in range(start, n):
                node = node.children.get(piece[end])
                if node is None:
                    break
                if node.token_id is None:
                    continue

                cost = 1 + best_cost[end + 1]
                current = best_next[start]
                if cost < best_cost[start] or (
                    cost == best_cost[start]
                    and current is not None
                    and node.token_id < current[2]
                ):
                    best_cost[start] = cost
                    best_next[start] = (end + 1, node.token or "", node.token_id)

            if best_next[start] is None:
                # Should not happen with a complete byte-level alphabet, but it
                # makes the API total for custom vocabularies.
                best_cost[start] = 1 + best_cost[start + 1]
                best_next[start] = (start + 1, self.unk_token, self.unk_id)

        tokens = []
        ids = []
        index = 0
        while index < n:
            next_step = best_next[index]
            if next_step is None:
                raise RuntimeError(f"No DP path found for piece {piece!r} at offset {index}")
            index, token, token_id = next_step
            tokens.append(token)
            ids.append(token_id)

        return Tokenization(tokens=tokens, ids=ids)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text).ids)

    def encode_batch(
        self,
        texts: list[str],
        *,
        num_workers: int = 1,
        chunksize: int = 64,
        start_method: str | None = None,
    ) -> list[Tokenization]:
        if num_workers <= 1:
            return [self.encode(text) for text in texts]

        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=get_process_context(start_method),
            initializer=_init_worker,
            initargs=(self.vocab, self.pretokenizer_mode, self.unk_token),
        ) as executor:
            return list(executor.map(_encode_worker, texts, chunksize=chunksize))

    def count_tokens_batch(
        self,
        texts: list[str],
        *,
        num_workers: int = 1,
        chunksize: int = 64,
        start_method: str | None = None,
    ) -> list[int]:
        if num_workers <= 1:
            return [self.count_tokens(text) for text in texts]

        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=get_process_context(start_method),
            initializer=_init_worker,
            initargs=(self.vocab, self.pretokenizer_mode, self.unk_token),
        ) as executor:
            return list(executor.map(_count_worker, texts, chunksize=chunksize))

    def save(self, path: str | Path) -> None:
        payload = {
            "model": "lp-dp",
            "vocab": self.vocab,
            "pretokenizer_mode": self.pretokenizer_mode,
            "unk_token": self.unk_token,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> "LpDpTokenizer":
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model") != "lp-dp":
            raise ValueError(f"Not an LP DP tokenizer file: {path}")
        return cls(
            payload["vocab"],
            pretokenizer_mode=payload["pretokenizer_mode"],
            unk_token=payload["unk_token"],
        )


def build_trie(vocab: list[str]) -> _TrieNode:
    root = _TrieNode(children={})
    for token_id, token in enumerate(vocab):
        node = root
        for char in token:
            node = node.children.setdefault(char, _TrieNode(children={}))
        node.token_id = token_id
        node.token = token
    return root


def get_process_context(start_method: str | None):
    if start_method is not None:
        return mp.get_context(start_method)
    if os.name == "posix" and "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


_WORKER_TOKENIZER: LpDpTokenizer | None = None


def _init_worker(vocab: list[str], pretokenizer_mode: str, unk_token: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = LpDpTokenizer(
        vocab,
        pretokenizer_mode=pretokenizer_mode,
        unk_token=unk_token,
    )


def _encode_worker(text: str) -> Tokenization:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("LP DP tokenizer worker was not initialized")
    return _WORKER_TOKENIZER.encode(text)


def _count_worker(text: str) -> int:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("LP DP tokenizer worker was not initialized")
    return _WORKER_TOKENIZER.count_tokens(text)
