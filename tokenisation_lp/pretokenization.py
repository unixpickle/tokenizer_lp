from __future__ import annotations

from tokenizers import Regex
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import ByteLevel, PreTokenizer, Sequence, Split


APERTUS_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
NANOCHAT_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)

SPLIT_PATTERNS = {
    "apertus": APERTUS_SPLIT_PATTERN,
    "nanochat": NANOCHAT_SPLIT_PATTERN,
}

DEFAULT_SPECIAL_TOKENS = ["<unk>", "<eos>"]
DEFAULT_UNK_TOKEN = "<unk>"


def build_pretokenizer(mode: str = "bytelevel") -> tuple[PreTokenizer, ByteLevelDecoder]:
    """Build a reusable pretokenizer/decoder pair.

    The LP and BPE trainers both operate on pretokenized strings. ByteLevel keeps
    every UTF-8 byte representable through a fixed 256-symbol alphabet.
    """

    normalized = mode.strip().lower()
    bytelevel = ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)

    if normalized in {"bytelevel", "split_bytelevel"}:
        return bytelevel, ByteLevelDecoder()

    if normalized in SPLIT_PATTERNS:
        return (
            Sequence(
                [
                    Split(
                        pattern=Regex(SPLIT_PATTERNS[normalized]),
                        behavior="isolated",
                        invert=False,
                    ),
                    ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
                ]
            ),
            ByteLevelDecoder(),
        )

    raise ValueError(
        f"Unsupported pretokenizer mode {mode!r}. "
        "Expected one of: bytelevel, split_bytelevel, apertus, nanochat."
    )


def byte_level_alphabet() -> list[str]:
    return sorted(ByteLevel.alphabet())


def pretokenize_text(text: str, pretokenizer: PreTokenizer) -> list[str]:
    return [piece for piece, _ in pretokenizer.pre_tokenize_str(text) if piece]

