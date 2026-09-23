"""Table structure data contract.

Separate from src/vision/schemas.py::TableStructure, which stays frozen and
untouched. That model records a recovered grid as rows of strings; it has no
cell identity, no spans, no per-cell bbox and no way to say "I am not sure".
Those are exactly the things this stage needs, and bolting them onto a frozen
model would force a change inside the vision freeze scope for reasons that have
nothing to do with vision.

Two rules run through every field here:

  raw and normalized text are stored separately. The raw form is the evidence
  of what the page says; the normalized form is what matching needs. Keeping
  only one of them throws away either the proof or the usability.

  uncertainty is a value, not an absence. A cell whose OCR block straddles two
  columns is recorded as assignment_uncertain rather than assigned to the
  nearer one, because a confidently wrong cell is worse than a flagged one.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# bbox is [x0, y0, x1, y1] in the ORIGINAL image's pixel coordinates, top-left
# origin. One convention everywhere: a mixed convention silently shifts every
# crop and every citation.
BBox = List[float]

ContinuationStatus = Literal[
    "same_page",
    "continuation_candidate",
    "continued_confirmed",
    "continuation_uncertain",
    "separate_table",
]

TableKind = Literal["grid_table", "borderless_table_candidate", "not_a_table"]
MergeStatus = Literal["none", "horizontal", "vertical", "uncertain"]
ParserSource = Literal["ocr", "vlm", "table_parser", "mixed"]


def deterministic_id(*parts: Any) -> str:
    """Stable id from structural position + content.

    Never a random UUID: re-parsing an unchanged page has to produce the same
    ids, or every stored reference breaks on each rebuild and two runs cannot
    be diffed.
    """
    payload = "|".join(str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class SourceBlockRef(BaseModel):
    """One OCR block that contributed text to a cell, and how much of it."""

    block_index: int
    text: str
    bbox: BBox
    confidence: float = 0.0
    # Share of the block's area lying inside the cell. Kept so a reviewer can
    # see why a straddling block was or was not accepted.
    coverage_ratio: float = 0.0


class TableCell(BaseModel):
    cell_id: str
    text_raw: str = ""
    text_normalized: str = ""
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    bbox: BBox = Field(default_factory=list)
    confidence: float = 0.0
    is_header: bool = False
    # Parent-to-child, e.g. ["参数", "功率"].
    header_path: List[str] = Field(default_factory=list)
    # Which header cells produced that path, so a value can point at where its
    # column name came from rather than asserting it.
    header_source_cell_ids: List[str] = Field(default_factory=list)
    source_blocks: List[SourceBlockRef] = Field(default_factory=list)
    assignment_uncertain: bool = False
    merge_status: MergeStatus = "none"
    merge_reasons: List[str] = Field(default_factory=list)

    @property
    def rowspan(self) -> int:
        return self.row_end - self.row_start + 1

    @property
    def colspan(self) -> int:
        return self.col_end - self.col_start + 1

    @property
    def is_empty(self) -> bool:
        return not self.text_raw.strip()


class ParsedTable(BaseModel):
    document_id: str
    table_id: str
    kind: TableKind = "grid_table"
    page_start: int
    page_end: int
    bbox: BBox = Field(default_factory=list)
    title: str = ""
    section_path: List[str] = Field(default_factory=list)
    cells: List[TableCell] = Field(default_factory=list)
    # Indices of rows that are headers rather than data.
    header_rows: List[int] = Field(default_factory=list)
    # Column holding the business entity, when one was identified. None rather
    # than 0: defaulting to the first column invents an entity for every table.
    entity_column: Optional[int] = None
    continuation_status: ContinuationStatus = "same_page"
    continuation_reasons: List[str] = Field(default_factory=list)
    structure_confidence: float = 0.0
    structure_uncertain: bool = False
    parser_source: ParserSource = "table_parser"
    source_hash: str = ""
    row_count: int = 0
    col_count: int = 0
    notes: List[str] = Field(default_factory=list)

    def cell_at(self, row: int, col: int) -> Optional[TableCell]:
        for cell in self.cells:
            if cell.row_start <= row <= cell.row_end and \
                    cell.col_start <= col <= cell.col_end:
                return cell
        return None

    def data_rows(self) -> List[int]:
        return [r for r in range(self.row_count) if r not in self.header_rows]


class ImageRegionRef(BaseModel):
    """A saved crop, tied back to the exact pixels it came from.

    Both the requested bbox and the bbox actually cropped are recorded: near a
    page edge padding gets clipped, and a reader comparing the crop against the
    citation needs to know which one they are looking at.
    """

    image_path: str
    source_image_sha256: str
    crop_sha256: str
    page: int
    bbox_original: BBox
    bbox_with_padding: BBox
    region_type: Literal["table", "row", "cell"]
    document_id: str
    table_id: Optional[str] = None
    row_id: Optional[str] = None
    cell_id: Optional[str] = None
    clipped: bool = False


class LocalVisionReviewRequest(BaseModel):
    """A request to look at one region again with a vision model.

    `status` is the whole point of this type. `requested_not_invoked` means the
    system decided a second look was warranted and did not take it — reporting
    that as a successful review would turn an open question into a fabricated
    confirmation.
    """

    request_id: str
    chunk_id: Optional[str] = None
    image_region_ref: Optional[ImageRegionRef] = None
    expected_fields: List[str] = Field(default_factory=list)
    existing_ocr_values: Dict[str, str] = Field(default_factory=dict)
    review_reason: str = ""
    status: Literal["not_required", "requested_not_invoked",
                    "cache_replay", "real_inference"] = "requested_not_invoked"
    result: Optional[Dict[str, Any]] = None
