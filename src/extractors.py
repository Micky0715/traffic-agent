from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Character folding
#
# Every mapping below is one character to one character, which is the whole
# point: matching happens on the folded text, and because folding never inserts
# or removes a character, a match offset in the folded string is the SAME
# offset in the text that was handed in. That is what lets a span point back at
# the original query instead of at some normalized rewrite of it.
#
# NFKC would fold more, but it is not length-preserving in general, so spans
# taken against it could not be mapped back.
# ---------------------------------------------------------------------------

_FOLD = {}
for _offset in range(26):
    _FOLD[0xFF21 + _offset] = chr(ord("A") + _offset)   # Ａ-Ｚ
    _FOLD[0xFF41 + _offset] = chr(ord("a") + _offset)   # ａ-ｚ
for _offset in range(10):
    _FOLD[0xFF10 + _offset] = chr(ord("0") + _offset)   # ０-９
for _dash in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFF0D\uFE63":
    _FOLD[ord(_dash)] = "-"
_FOLD[0x3000] = " "                                      # ideographic space


def fold_for_match(text: str) -> str:
    """Length-preserving fold. len(fold_for_match(t)) == len(t) always."""
    return text.translate(_FOLD)


# ---------------------------------------------------------------------------
# Patterns
#
# `\b` cannot be used at either end of these. Python's re treats CJK as \w, so
# in "查JTG D81-2017中的要求" there is no word boundary before J or after 7, and
# the whole pattern fails — while the same code surrounded by spaces matches.
# Chinese queries put a character hard against the code almost every time, so
# in practice these patterns matched almost nothing.
#
# Worse than not matching: in "TB10621-2014对…" the trailing \b failed and the
# regex BACKTRACKED, producing a wrong normalization ("TB 1-0621"). A missed
# match is a gap; a backtracked one corrupts the value.
#
# The boundary these patterns actually want is "not part of a longer
# alphanumeric token", stated directly as ASCII lookarounds.
# ---------------------------------------------------------------------------

_NOT_CODE_BEFORE = r"(?<![A-Za-z0-9])"
_NOT_CODE_AFTER = r"(?![A-Za-z0-9])"

ASSET_PATTERN = re.compile(
    r"(?:[A-Z]\d{1,3}(?:号)?(?:风机|水泵|泵|扶梯|屏蔽门|信号机|设备)?|"
    r"\d+号(?:风机|水泵|泵|扶梯|屏蔽门|信号机)|"
    r"(?:天河站|体育西路站|广州南站|珠江新城站)[A-Z]?\d{0,3}(?:风机|水泵|泵|扶梯|屏蔽门)?)"
)
REG_PATTERN = re.compile(
    _NOT_CODE_BEFORE + r"(JTG|TB|GB|CJJ)\s*([A-Z]?\d+[A-Z]?)\s*-\s*(\d{4})" + _NOT_CODE_AFTER,
    re.I)
DRAWING_PATTERN = re.compile(
    _NOT_CODE_BEFORE + r"(?:DWG|TUN|ELEC|MEP|FAN)-?[A-Z0-9-]{2,}" + _NOT_CODE_AFTER, re.I)
TIME_PATTERNS = [
    re.compile(r"近|最近|过去"),
    re.compile(r"\d+\s*(?:小时|天|周|月|年)"),
    re.compile(r"今天|昨日|昨天|本周|上周|本月|上月|当前|现在"),
    re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"),
]
METRIC_KEYWORDS = ["告警次数", "报警次数", "故障次数", "振动值", "温度", "电流", "完成率", "数量", "多少", "趋势", "平均值"]


@dataclass(frozen=True)
class CodeMention:
    """One regulation or drawing number found in a text.

    raw_value is exactly the characters that appeared; normalized_value is the
    canonical form. Both are kept because the raw form is the evidence of what
    the document actually said, and the canonical form is what downstream
    matching needs — collapsing them would throw away one or the other.

    start/end index the text that was PASSED IN. Callers that pass the original
    query get spans into the original query; callers that pass normalized text
    get spans into that. Folding is length-preserving precisely so this holds.
    """

    kind: str            # "regulation" | "drawing"
    raw_value: str
    normalized_value: str
    start: int
    end: int

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)


def _canonical_regulation(match: re.Match) -> str:
    prefix, middle, year = match.group(1), match.group(2), match.group(3)
    return f"{prefix.upper()} {middle.upper()}-{year}"


def find_code_mentions(text: str) -> List[CodeMention]:
    """Regulation and drawing numbers, with spans into `text`."""
    folded = fold_for_match(text)
    mentions: List[CodeMention] = []
    for match in REG_PATTERN.finditer(folded):
        mentions.append(CodeMention(
            kind="regulation", raw_value=text[match.start():match.end()],
            normalized_value=_canonical_regulation(match),
            start=match.start(), end=match.end()))
    for match in DRAWING_PATTERN.finditer(folded):
        mentions.append(CodeMention(
            kind="drawing", raw_value=text[match.start():match.end()],
            normalized_value=match.group(0).upper(),
            start=match.start(), end=match.end()))
    return sorted(mentions, key=lambda m: m.start)


def first_match(pattern: re.Pattern[str], text: str) -> Optional[str]:
    """Search the folded text so full-width and dash variants are found, but
    return the ORIGINAL characters, not the folded ones."""
    m = pattern.search(fold_for_match(text))
    return text[m.start():m.end()] if m else None


CHINESE_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _unique(items):
    return list(dict.fromkeys(items))


def extract_slots(text: str) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}
    folded = fold_for_match(text)

    # Codes first: an equipment id that lies INSIDE a regulation or drawing
    # number is not an equipment id. The D81 in "JTG D81-2017" and the A13 in
    # "FAN-A13-02" are fragments of a longer code.
    mentions = find_code_mentions(text)
    regulation = next((m for m in mentions if m.kind == "regulation"), None)
    drawing = next((m for m in mentions if m.kind == "drawing"), None)

    # Exclusion is by CHARACTER OVERLAP with a specific mention, never by
    # deleting every occurrence of the same string: "查JTG D81-2017，再看D81风机"
    # contains one fragment and one genuine mention of the same characters, and
    # a global blacklist would drop both.
    occupied = [m.span for m in mentions]
    assets = _unique([
        m.group(0) for m in ASSET_PATTERN.finditer(folded)
        if not any(m.start() < end and start < m.end() for start, end in occupied)
    ])

    # Prefer explicit equipment ids over a station-only match when both appear.
    explicit_assets = [x for x in assets if re.search(r"[A-Z]\d|\d+号", x)]
    selected_assets = explicit_assets or assets
    asset = selected_assets if len(selected_assets) > 1 else (selected_assets[0] if selected_assets else None)
    if asset:
        slots["asset_id"] = asset
    if regulation:
        slots["regulation_code"] = regulation.normalized_value
    if drawing:
        slots["drawing_no"] = drawing.normalized_value

    absolute_range = re.search(r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\s*(?:至|到|~|～|-)\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})", text)
    duration = re.search(r"(?:近|最近|过去)?\s*(\d+|[一二两三四五六七八九十])\s*(小时|天|周|月|年)", text)
    if duration:
        raw_num = duration.group(1)
        num = CHINESE_NUM.get(raw_num, raw_num)
        slots["time_range"] = f"最近{num}{duration.group(2)}"
    elif absolute_range:
        slots["time_range"] = f"{absolute_range.group(1)}至{absolute_range.group(2)}"
    else:
        tokens = [token for token in ["今天", "昨天", "昨日", "本周", "上周", "本月", "上月", "当前", "现在"] if token in text]
        tokens = ["昨天" if x == "昨日" else x for x in tokens]
        tokens = _unique(tokens)
        if len(tokens) > 1:
            slots["time_range"] = tokens
        elif tokens:
            slots["time_range"] = tokens[0]

    for metric in METRIC_KEYWORDS:
        if metric in text:
            slots["metric"] = metric
            break

    threshold = re.search(r"(?:超过|高于|大于|低于|小于)\s*(\d+(?:\.\d+)?)\s*([a-zA-Z/·%]+)?", text)
    if threshold:
        slots["threshold"] = threshold.group(1) + (threshold.group(2) or "")
    return slots


def has_time_signal(text: str) -> bool:
    return any(p.search(text) for p in TIME_PATTERNS)
