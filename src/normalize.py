from __future__ import annotations

import re

from src.extractors import fold_for_match

# Kept for callers that import it. The authoritative fold now lives in
# extractors.fold_for_match, which also covers full-width forms.
DASHES = "‐‑‒–—―−－﹣"


def normalize_regulation_code(text: str) -> str:
    r"""Canonicalize regulation codes to "<PREFIX> <MIDDLE>-<YEAR>".

    The `\b` that used to bound this pattern was the same bug as in
    extractors: Python treats CJK as \w, so "TB10621-2014对…" failed at the
    trailing boundary. It did not merely fail to match — the regex backtracked
    and rewrote the code as "TB 1-0621-2014", so normalization CORRUPTED the
    value it was meant to clean. A missed match is a gap; a corrupted one is
    worse, because everything downstream then trusts it.
    """
    text = fold_for_match(text).upper()
    text = re.sub(
        r"(?<![A-Za-z0-9])(JTG|TB|GB|CJJ)\s*([A-Z]?\d+[A-Z]?)\s*-?\s*(\d{4})(?![A-Za-z0-9])",
        r"\1 \2-\3", text)
    return text


def normalize_text(text: str) -> str:
    text = text.strip()
    text = normalize_regulation_code(text)
    text = re.sub(r"[，,]+", "，", text)
    text = re.sub(r"[；;]+", "；", text)
    text = re.sub(r"\s+", " ", text)
    return text
