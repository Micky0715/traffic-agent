from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from src.vision.schemas import RegulationClause, RegulationMetadata

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}

CHAPTER_RE = re.compile(r"第([一二三四五六七八九十百千0-9]+)章")
SECTION_RE = re.compile(r"第([一二三四五六七八九十百千0-9]+)节")
ARTICLE_RE = re.compile(r"第([一二三四五六七八九十百千0-9]+)条")
CLAUSE_RE = re.compile(r"第([一二三四五六七八九十百千0-9]+)款")
PAREN_CLAUSE_RE = re.compile(r"[（(]([一二三四五六七八九十]+)[）)]")
TITLE_RE = re.compile(r"《([^》]+)》")
DATE_RE = re.compile(r"(\d{4}年\d{1,2}月\d{1,2}日)")


def chinese_to_arabic(cn: str) -> Optional[int]:
    """Bounded converter for the range regulation article numbers actually
    use (roughly 1-999). Returns None on anything it can't parse rather than
    guessing — an unrecognized numeral is worse to silently mis-convert
    than to admit uncertainty about."""
    if not cn:
        return None
    if cn.isdigit():
        return int(cn)
    section = 0
    num = 0
    for ch in cn:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            section += (num or 1) * _CN_UNITS[ch]
            num = 0
        else:
            return None
    return section + num


def _is_noise_line(line: str, line_counts: Counter) -> bool:
    """Heuristic header/footer filter: short lines with no structural marker
    that repeat 2+ times anywhere in the document (a real page header/footer
    would). There is no page-boundary information in a plain-text input, so
    this is a same-document-repetition heuristic, not true per-page
    detection — documented limitation, see interview/ocr_bad_cases.md."""
    if len(line) > 20:
        return False
    if any(p.search(line) for p in (CHAPTER_RE, SECTION_RE, ARTICLE_RE, CLAUSE_RE, PAREN_CLAUSE_RE)):
        return False
    return line_counts[line] >= 2


def parse_regulation_text(text: str) -> RegulationMetadata:
    lines = [l.strip() for l in text.splitlines()]
    line_counts = Counter(l for l in lines if l)

    title_match = TITLE_RE.search(text)
    date_match = DATE_RE.search(text)
    issuer = None
    for line in lines[:10]:
        if "发布" in line or "批准" in line:
            issuer = line
            break

    clauses: List[RegulationClause] = []
    warnings: List[str] = []
    chapter = section = article = clause = None
    buffer: List[str] = []
    start_line = 0
    last_article_num: Optional[int] = None
    last_article_raw: Optional[str] = None

    def flush() -> None:
        if buffer:
            body = "".join(buffer).strip()
            if body:
                clauses.append(RegulationClause(
                    chapter=chapter, section=section, article=article, clause=clause,
                    text=body, line_number=start_line,
                ))

    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or _is_noise_line(line, line_counts):
            continue

        m_chapter = CHAPTER_RE.search(line)
        m_section = None if m_chapter else SECTION_RE.search(line)
        m_article = None if (m_chapter or m_section) else ARTICLE_RE.search(line)
        m_clause = None
        if not (m_chapter or m_section or m_article):
            m_clause = CLAUSE_RE.search(line) or PAREN_CLAUSE_RE.search(line)

        if m_chapter:
            flush()
            chapter, section, article, clause = f"第{m_chapter.group(1)}章", None, None, None
            start_line, buffer = i, [line]
            continue
        if m_section:
            flush()
            section, article, clause = f"第{m_section.group(1)}节", None, None
            start_line, buffer = i, [line]
            continue
        if m_article:
            flush()
            raw_num = m_article.group(1)
            article, clause = f"第{raw_num}条", None
            num = chinese_to_arabic(raw_num)
            if num is not None and last_article_num is not None and num != last_article_num + 1:
                warnings.append(f"第{last_article_raw}条后直接出现第{raw_num}条，中间条款可能缺失或编号不连续")
            if num is not None:
                last_article_num, last_article_raw = num, raw_num
            start_line, buffer = i, [line]
            continue
        if m_clause:
            flush()
            raw_num = m_clause.group(1)
            clause = f"第{raw_num}款" if CLAUSE_RE.search(line) else f"（{raw_num}）"
            start_line, buffer = i, [line]
            continue

        buffer.append(line)

    flush()

    return RegulationMetadata(
        document_name=title_match.group(1) if title_match else None,
        issuer=issuer,
        effective_date=date_match.group(1) if date_match else None,
        clauses=clauses,
        warnings=warnings,
    )
