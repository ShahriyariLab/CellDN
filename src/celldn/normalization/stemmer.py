"""Plural normalization helpers used by CellDN NEN."""

from __future__ import annotations

import re
from typing import Iterable


_TOKEN_FINDER = re.compile(r"[^\W_]+|[^\w\s]|_|,", re.UNICODE)


def split_tokens(value: object) -> list[str]:
    """Split text into the original CellDN token representation."""

    return _TOKEN_FINDER.findall(str(value))


def _replace_tail(word: str, suffix: str, replacement: str) -> str:
    return word[: -len(suffix)] + replacement


def normalize_token(value: object) -> str:


    word = str(value)
    if not word.endswith("s"):
        return word

    if word.endswith("viruses"):
        return _replace_tail(word, "uses", "us")

    if word.endswith("ies"):
        if not word.endswith(("eies", "aies")):
            return _replace_tail(word, "ies", "y")

    if word.endswith("es"):
        if not word.endswith(("aes", "ees", "oes")):
            if word.endswith("sses"):
                return _replace_tail(word, "es", "")
            return _replace_tail(word, "es", "e")

    if word.endswith(("us", "ss")):
        return word

    return _replace_tail(word, "s", "")


def plural_normalize_text(value: object) -> str:
    """Apply token-level plural normalization to a text value."""

    return " ".join(normalize_token(part) for part in split_tokens(value))


__all__ = [
    "split_tokens",
    "normalize_token",
    "plural_normalize_text",
]
