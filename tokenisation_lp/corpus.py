from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def iter_text_files(data_dir: str | Path) -> Iterator[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Text directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory of text files: {root}")

    for path in sorted(root.rglob("*")):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts):
            yield path


def iter_texts(data_dir: str | Path) -> Iterator[str]:
    for path in iter_text_files(data_dir):
        yield path.read_text(encoding="utf-8", errors="replace")


def load_texts(data_dir: str | Path) -> list[str]:
    texts = [text for text in iter_texts(data_dir) if text]
    if not texts:
        raise ValueError(f"No non-empty text files found in: {data_dir}")
    return texts

