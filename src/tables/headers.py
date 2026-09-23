"""Multi-level header expansion.

A two-row header

    | 设备 | 参数 |      | 位置 |
    |      | 功率 | 风量 |      |

has to become

    设备 / 参数.功率 / 参数.风量 / 位置

The blank under 参数 in row 1 is a merged cell spanning two columns, so both
sub-columns inherit 参数. The blank under 位置 in row 2 is not: 位置 has no
sub-header, and its column's path is just 位置.

Getting that distinction wrong is not cosmetic. An earlier version of this
logic carried a blank's value leftward without checking whether the columns
shared a parent, and 备注 came out as 备注.风量 — after which a query for 风量
matches a remarks cell. That case is pinned as a regression test.

The rule: a blank inherits from its left neighbour ONLY while both columns have
the same ancestry above the current level. Crossing a top-level boundary stops
the inheritance, because the columns are then under different parents and
nothing links them.

Repeated text is deliberately not treated as evidence of a merge. Two adjacent
columns both headed 功率 may be two separate measurements, and collapsing them
would merge two different quantities into one.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from src.tables.config import HeaderConfig
from src.tables.schemas import ParsedTable, TableCell


def _row_cells(table: ParsedTable, row: int) -> List[Optional[TableCell]]:
    return [table.cell_at(row, col) for col in range(table.col_count)]


def detect_header_rows(table: ParsedTable, cfg: HeaderConfig,
                       declared: Optional[int] = None) -> List[int]:
    """Which rows are headers.

    The caller usually knows; when it does not, a weak heuristic is used and
    the result is a hint, not an assertion. Guessing header depth from geometry
    alone produced a real bug once already (a title block's first row silently
    became a header and vanished from the data).
    """
    if declared is not None:
        return list(range(min(declared, table.row_count)))
    if table.row_count == 0:
        return []
    header_rows: List[int] = []
    for row in range(min(cfg.max_header_rows, table.row_count)):
        cells = [c for c in _row_cells(table, row) if c and c.text_normalized]
        if not cells:
            continue
        numeric = sum(1 for c in cells
                      if any(ch.isdigit() for ch in c.text_normalized))
        if numeric / len(cells) >= cfg.numeric_ratio_for_data_row:
            break
        header_rows.append(row)
    return header_rows or list(range(min(cfg.default_header_rows, table.row_count)))


def expand_header_paths(table: ParsedTable,
                        header_rows: Sequence[int]) -> Dict[int, List[str]]:
    """Column index -> header path, parent first.

    Also records, on each header cell, which columns it covers, so a data cell
    can name the exact cells its path came from rather than asserting a path
    out of nowhere.
    """
    paths: Dict[int, List[str]] = {col: [] for col in range(table.col_count)}
    sources: Dict[int, List[str]] = {col: [] for col in range(table.col_count)}

    for level, row in enumerate(header_rows):
        carried_text = ""
        carried_source = ""
        carried_ancestry: List[str] = []
        for col in range(table.col_count):
            cell = table.cell_at(row, col)
            if cell is not None:
                cell.is_header = True
            text = cell.text_normalized if cell else ""
            ancestry = paths[col][:level]

            if text:
                carried_text, carried_source, carried_ancestry = \
                    text, (cell.cell_id if cell else ""), ancestry
            elif carried_text and ancestry == carried_ancestry:
                # Blank inside the same merged span: inherit.
                pass
            else:
                # Different parent above, or nothing to inherit. Crossing a
                # top-level boundary is where 备注 would otherwise pick up the
                # neighbouring 风量.
                carried_text, carried_source, carried_ancestry = "", "", ancestry

            if carried_text:
                paths[col].append(carried_text)
                if carried_source:
                    sources[col].append(carried_source)

    for col in range(table.col_count):
        for row in table.data_rows():
            cell = table.cell_at(row, col)
            if cell is None:
                continue
            cell.header_path = list(paths[col])
            cell.header_source_cell_ids = list(dict.fromkeys(sources[col]))

    table.header_rows = list(header_rows)
    return paths


def header_path_strings(paths: Dict[int, List[str]]) -> List[str]:
    return [".".join(paths[col]) if paths.get(col) else f"第{col + 1}列"
            for col in sorted(paths)]


def looks_like_repeated_header(table: ParsedTable, row: int,
                               header_rows: Sequence[int]) -> bool:
    """A data row identical to a header row is a repeated header.

    Matters on a continued table: the header reprinted at the top of page two
    is not a device.
    """
    candidate = [(table.cell_at(row, col).text_normalized
                  if table.cell_at(row, col) else "")
                 for col in range(table.col_count)]
    if not any(candidate):
        return False
    for header_row in header_rows:
        reference = [(table.cell_at(header_row, col).text_normalized
                      if table.cell_at(header_row, col) else "")
                     for col in range(table.col_count)]
        if reference == candidate:
            return True
    return False
