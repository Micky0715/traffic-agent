"""Cross-page table continuation.

Conservative on purpose. Wrongly splitting a continued table loses continuity —
annoying, recoverable, visible. Wrongly merging two tables invents rows that
belong to a different table, and nothing downstream can tell that happened. The
two errors are not symmetric, so the thresholds are not either.

Matching column counts is explicitly not sufficient: two unrelated 3-column
tables on facing pages would merge on that alone.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from src.tables.config import ContinuationConfig
from src.tables.schemas import ParsedTable


def _horizontal_extent_compatible(a: ParsedTable, b: ParsedTable,
                                  tolerance: int) -> bool:
    """A continued table keeps roughly the same left and right edges.

    A table starting at a different x is a different table, however similar its
    columns look.
    """
    if not a.bbox or not b.bbox:
        return False
    return (abs(a.bbox[0] - b.bbox[0]) <= tolerance and
            abs(a.bbox[2] - b.bbox[2]) <= tolerance)


def assess_continuation(
    previous: ParsedTable,
    current: ParsedTable,
    cfg: ContinuationConfig,
    *,
    previous_header_paths: Optional[Sequence[str]] = None,
    current_header_paths: Optional[Sequence[str]] = None,
    repeated_header_on_next_page: bool = False,
    continuation_marker: bool = False,
    previous_unterminated: bool = False,
) -> Dict[str, Any]:
    """Decide whether `current` continues `previous`, and say why or why not."""
    signals = {
        "adjacent_pages": current.page_start - previous.page_end == 1,
        "same_document": previous.document_id == current.document_id,
        "column_count_compatible": previous.col_count == current.col_count,
        "header_path_compatible": (
            list(previous_header_paths or []) == list(current_header_paths or [])
            if (previous_header_paths or current_header_paths) else False),
        "horizontal_extent_compatible": _horizontal_extent_compatible(
            previous, current, cfg.horizontal_extent_tolerance_px),
        # Absence of section information is not evidence of sameness. Two
        # tables that both record no section match trivially, and letting that
        # count as support would merge on shape alone — the exact failure the
        # required signals exist to prevent.
        "same_section": bool(previous.section_path) and
        previous.section_path == current.section_path,
        "repeated_header_on_next_page": repeated_header_on_next_page,
        "continuation_marker": continuation_marker,
        "previous_unterminated": previous_unterminated,
    }

    satisfied = {name for name, value in signals.items() if value}
    missing_required = [f"not_{name}" for name in cfg.required_signals
                        if name not in satisfied]
    supporting = len(satisfied & set(cfg.supporting_signals))

    if missing_required:
        status = "separate_table"
        reasons = missing_required
    elif supporting >= cfg.min_supporting:
        status = "continued_confirmed"
        reasons = [f"supported_by_{name}" for name in
                   sorted(satisfied & set(cfg.supporting_signals))]
    else:
        # Shape matches but nothing says the table actually carried over. Left
        # separate and flagged for a human rather than merged on a guess.
        status = "continuation_uncertain"
        reasons = ["required_signals_met_but_no_supporting_evidence"]

    return {"status": status, "signals": signals, "reasons": reasons,
            "supporting_count": supporting}


def apply_continuation(previous: ParsedTable, current: ParsedTable,
                       assessment: Dict[str, Any]) -> None:
    """Record the outcome on both tables. Never rewrites their cells."""
    status = assessment["status"]
    current.continuation_status = (
        "continuation_candidate" if status == "continued_confirmed" else status)
    current.continuation_reasons = list(assessment["reasons"])
    if status in {"continuation_uncertain", "continuation_candidate"}:
        current.structure_uncertain = True
    if status == "continued_confirmed":
        previous.continuation_status = "continuation_candidate"
        previous.continuation_reasons = list(assessment["reasons"])
