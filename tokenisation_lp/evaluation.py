from __future__ import annotations

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompressionStats:
    name: str
    files: int
    bytes: int
    chars: int
    tokens: int

    @property
    def bytes_per_token(self) -> float:
        return self.bytes / self.tokens if self.tokens else 0.0

    @property
    def chars_per_token(self) -> float:
        return self.chars / self.tokens if self.tokens else 0.0

    @property
    def tokens_per_byte(self) -> float:
        return self.tokens / self.bytes if self.bytes else 0.0


def evaluate_texts(name: str, tokenizer, texts: list[str], *, num_workers: int = 1) -> CompressionStats:
    total_bytes = 0
    total_chars = 0

    for text in texts:
        total_bytes += len(text.encode("utf-8"))
        total_chars += len(text)

    if hasattr(tokenizer, "count_tokens_batch"):
        token_counts = tokenizer.count_tokens_batch(texts, num_workers=num_workers)
    else:
        token_counts = [len(tokenizer.encode(text).ids) for text in texts]
    total_tokens = sum(token_counts)

    stats = CompressionStats(
        name=name,
        files=len(texts),
        bytes=total_bytes,
        chars=total_chars,
        tokens=total_tokens,
    )
    log_compression_stats(stats)
    return stats


def log_compression_stats(stats: CompressionStats) -> None:
    LOGGER.info(
        "%s compression: files=%d bytes=%d chars=%d tokens=%d "
        "bytes/token=%.4f chars/token=%.4f tokens/byte=%.6f",
        stats.name,
        stats.files,
        stats.bytes,
        stats.chars,
        stats.tokens,
        stats.bytes_per_token,
        stats.chars_per_token,
        stats.tokens_per_byte,
    )
