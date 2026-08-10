from __future__ import annotations

import re


DASHES = "‐‑‒–—―−"


def normalize_regulation_code(text: str) -> str:
    text = text.upper()
    text = re.sub(f"[{DASHES}]", "-", text)
    # JTGD81-2017 / JTG D81—2017 -> JTG D81-2017
    text = re.sub(r"\b(JTG|TB|GB|CJJ)\s*([A-Z]?\d+[A-Z]?)\s*-?\s*(\d{4})\b", r"\1 \2-\3", text)
    return text


def normalize_text(text: str) -> str:
    text = text.strip()
    text = normalize_regulation_code(text)
    text = re.sub(r"[，,]+", "，", text)
    text = re.sub(r"[；;]+", "；", text)
    text = re.sub(r"\s+", " ", text)
    return text
