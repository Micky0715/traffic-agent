"""Finding a region worth re-reading, in six graded steps.

The baseline audit measured the problem this module exists for: 9 of 11 review
targets could not be cropped, and every one of those 9 sat on a page where a
table HAD been recovered — table bbox, cell bboxes, table region and
neighbouring OCR anchors all present. The information was there; one function
that looked only for an exact label text was the only thing consuming it.

The grading is the substance. A crop of a whole title block and a crop of one
value cell are both "a region", and calling both of them a successful
localisation is how a report ends up claiming precision it does not have. So
every region carries the strategy that produced it and a trust level, and
`answer_eligible` is false for anything coarser than a located field.

Nothing here is allowed to consult a file name, a case id, an absolute
coordinate or an expected value. A region chosen by knowing the answer can only
ever confirm the answer.
"""
from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.vision.review_region_config import ReviewRegionConfig

# Strategies, coarsest last.
EXACT_LABEL_RIGHT = "exact_label_right"
ALIAS_LABEL_RIGHT = "alias_label_right"
TABLE_VALUE_CELL = "table_value_cell"
STRUCTURAL_ROW_INFERENCE = "structural_row_inference"
TABLE_OR_TITLE_BLOCK = "table_or_title_block"
FULL_PAGE_DIAGNOSTIC = "full_page_diagnostic"
UNAVAILABLE = "unavailable"

PRECISE, STRUCTURAL, CONTEXTUAL, DIAGNOSTIC = (
    "precise", "structural", "contextual", "diagnostic")

STRATEGY_TRUST = {
    EXACT_LABEL_RIGHT: PRECISE,
    ALIAS_LABEL_RIGHT: STRUCTURAL,        # capped: a fuzzy label is not precise
    TABLE_VALUE_CELL: STRUCTURAL,
    STRUCTURAL_ROW_INFERENCE: STRUCTURAL,
    TABLE_OR_TITLE_BLOCK: CONTEXTUAL,
    FULL_PAGE_DIAGNOSTIC: DIAGNOSTIC,
}

TRUST_RANK = {PRECISE: 0, STRUCTURAL: 1, CONTEXTUAL: 2, DIAGNOSTIC: 3}

# Only a located field may feed an automatic answer. Everything coarser is a
# region someone still has to interpret.
ANSWER_ELIGIBLE_TRUST = {PRECISE}

# Why nothing could be produced. A code, not a sentence, so the report can
# count them.
R_IMAGE_MISSING = "image_missing_or_unreadable"
R_NO_VALID_BBOX = "no_valid_bbox_survived_clipping"
R_NO_STRUCTURE = "no_trustworthy_structural_region"
R_DIAGNOSTIC_DISABLED = "full_page_diagnostic_disabled_by_config"
R_AMBIGUOUS_ALIAS = "alias_match_ambiguous_or_below_threshold"


@dataclass
class ResolvedReviewRegion:
    request_id: str
    document_id: str
    page_no: int
    target_fields: List[str]
    bbox: List[float]
    strategy: str
    trust_level: str
    source_anchors: List[str] = field(default_factory=list)
    area_ratio: float = 0.0
    reasons: List[str] = field(default_factory=list)
    localization_uncertain: bool = True
    answer_eligible: bool = False
    # Set only for a downscaled diagnostic image.
    scale: float = 1.0
    unavailable_reason: Optional[str] = None
    alias_match: Optional[Dict] = None

    @property
    def available(self) -> bool:
        return self.strategy != UNAVAILABLE and bool(self.bbox)

    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id, "document_id": self.document_id,
            "page_no": self.page_no, "target_fields": list(self.target_fields),
            "bbox": [float(v) for v in self.bbox], "strategy": self.strategy,
            "trust_level": self.trust_level,
            "source_anchors": list(self.source_anchors),
            "area_ratio": round(self.area_ratio, 6),
            "reasons": list(self.reasons),
            "localization_uncertain": self.localization_uncertain,
            "answer_eligible": self.answer_eligible,
            "scale": self.scale,
            "unavailable_reason": self.unavailable_reason,
            "alias_match": self.alias_match,
        }


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _valid(bbox: Sequence[float]) -> bool:
    return (len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1])


def clip_to_image(bbox: Sequence[float], width: int, height: int) -> List[float]:
    """Never negative, never zero-area, never outside the page."""
    x0, x1 = sorted((float(bbox[0]), float(bbox[2])))
    y0, y1 = sorted((float(bbox[1]), float(bbox[3])))
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(width), x1), min(float(height), y1)
    if x1 <= x0 or y1 <= y0:
        return []
    return [x0, y0, x1, y1]


def pad(bbox: Sequence[float], cfg: ReviewRegionConfig,
        width: int, height: int) -> List[float]:
    """Padding proportional to the region's own short side.

    A fixed pixel margin that suits a 40px cell swallows the neighbouring rows
    of a title block, and one that suits a title block barely moves a cell edge.
    """
    short = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
    amount = min(max(short * cfg.crop_padding_ratio, cfg.min_crop_padding_px),
                 cfg.max_crop_padding_px)
    return clip_to_image(
        [bbox[0] - amount, bbox[1] - amount, bbox[2] + amount, bbox[3] + amount],
        width, height)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    if not _valid(a) or not _valid(b):
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def area_ratio(bbox: Sequence[float], width: int, height: int) -> float:
    if not _valid(bbox) or width <= 0 or height <= 0:
        return 0.0
    return ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (width * height)


def _same_row(a: Sequence[float], b: Sequence[float], ratio: float) -> bool:
    """Vertical overlap relative to the SHORTER box, so a tall OCR box does not
    swallow every line it brushes against."""
    top, bottom = max(a[1], b[1]), min(a[3], b[3])
    if bottom <= top:
        return False
    shorter = min(a[3] - a[1], b[3] - b[1])
    return shorter > 0 and (bottom - top) / shorter >= ratio


# ---------------------------------------------------------------------------
# L1 / L2 — label-anchored
# ---------------------------------------------------------------------------

def _block_text(block) -> str:
    return (getattr(block, "text", "") or "").strip()


def _block_bbox(block) -> List[float]:
    bbox = list(getattr(block, "bbox", []) or [])
    return [float(v) for v in bbox] if len(bbox) == 4 else []


def find_exact_label(field_name: str, blocks: Sequence) -> Optional[Tuple[str, List[float]]]:
    for block in blocks:
        text, bbox = _block_text(block), _block_bbox(block)
        if field_name and field_name in text and _valid(bbox):
            return text, bbox
    return None


def find_alias_label(
    field_name: str, blocks: Sequence, cfg: ReviewRegionConfig,
) -> Tuple[Optional[Tuple[str, List[float]]], Optional[Dict]]:
    """A damaged label may stand in for a field ONLY if one candidate wins.

    Two candidates at similar scores is not a near miss, it is a question about
    which field this is — and answering it by taking the higher float would let
    a value be filed under the wrong name with no trace.
    """
    if not cfg.enable_alias_match:
        return None, None
    aliases = cfg.field_aliases.get(field_name, [field_name])

    scored: List[Tuple[float, str, List[float], str]] = []
    for block in blocks:
        text, bbox = _block_text(block), _block_bbox(block)
        if not text or not _valid(bbox):
            continue
        best = max(
            ((difflib.SequenceMatcher(None, alias, text).ratio(), alias)
             for alias in aliases), key=lambda pair: pair[0])
        if best[0] >= cfg.alias_similarity_threshold:
            scored.append((best[0], text, bbox, best[1]))

    if not scored:
        return None, None
    scored.sort(key=lambda row: -row[0])
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-6:
        return None, {"ambiguous": True,
                      "candidates": [row[1] for row in scored[:3]],
                      "score": scored[0][0]}
    score, text, bbox, alias = scored[0]
    return (text, bbox), {"matched_text": text, "matched_alias": alias,
                          "score": round(score, 4), "ambiguous": False}


def value_region_right_of(
    label_bbox: Sequence[float], cfg: ReviewRegionConfig,
    width: int, height: int, cells: Sequence = (),
) -> List[float]:
    """Where the value for a label sits.

    Prefers a grid cell to the right on the same row — a real boundary beats a
    guessed extent. Falls back to a fixed rightward extension only when there
    is no grid to consult.
    """
    same_row = [c for c in cells
                if _valid(getattr(c, "bbox", []) or [])
                and _same_row(c.bbox, label_bbox, cfg.same_row_overlap_ratio)
                and c.bbox[0] >= label_bbox[2] - 1]
    if same_row:
        cell = min(same_row, key=lambda c: c.bbox[0])
        return clip_to_image(list(cell.bbox), width, height)
    return clip_to_image(
        [label_bbox[0], label_bbox[1],
         min(float(width), label_bbox[2] + width * cfg.label_right_extension_ratio),
         label_bbox[3]], width, height)


# ---------------------------------------------------------------------------
# L3 — key/value table row
# ---------------------------------------------------------------------------

def infer_key_value_row(
    field_name: str,
    table,
    cfg: ReviewRegionConfig,
    identified_rows: Dict[int, str],
) -> Tuple[Optional[int], List[str]]:
    """Which row of a confirmed key/value table holds `field_name`.

    Two independent routes, both structural:

      * the row's own left cell already reads as this field (or an alias);
      * the rows above and below are both identified, and the configured field
        order says this field belongs between them.

    The second is what recovers a label destroyed beyond recognition WITHOUT
    hard-coding the damaged spelling: it is the neighbours that place the row,
    not the ruined text. Returns (None, reasons) when the row is not uniquely
    determined — a probable row is not a row.
    """
    anchors: List[str] = []
    order = cfg.title_block_field_order
    if field_name not in order:
        return None, ["field is not in the configured title-block order"]

    index = order.index(field_name)
    before = order[:index]
    after = order[index + 1:]

    rows = sorted(identified_rows)
    above = [r for r in rows if identified_rows[r] in before]
    below = [r for r in rows if identified_rows[r] in after]
    if not above or not below:
        return None, ["no identified field both above and below this one"]

    low, high = max(above), min(below)
    candidates = [r for r in range(low + 1, high)
                  if r not in identified_rows]
    if len(candidates) != 1:
        return None, [f"{len(candidates)} unidentified rows between "
                      f"{identified_rows[low]!r} and {identified_rows[high]!r}; "
                      f"the row is not uniquely determined"]
    anchors = [f"row{low}:{identified_rows[low]}", f"row{high}:{identified_rows[high]}"]
    return candidates[0], anchors


def identify_key_value_rows(
    table, cfg: ReviewRegionConfig,
) -> Dict[int, str]:
    """Map row index -> field name, for rows whose left cell is recognisable.

    Uses the SAME conservative fuzzy matcher and threshold as L2, not exact
    substring matching. A key/value table whose labels are intact does not need
    L3 at all — L1 already found them in the OCR blocks. L3 exists for the case
    where those labels are damaged, so matching them exactly made the whole
    level unreachable: measured on FAN-A24-01, whose left column OCRs as
    'N' / 'ne' / '电r编号' / 'E银行 制编', substring matching identified 0 of 4
    rows.

    A row is claimed only when ONE field wins outright. '制编' scores 0.364
    against every configured alias and is therefore left unidentified — which
    is the intended outcome, not a gap to be patched with a special case.
    """
    scored: Dict[int, List[Tuple[float, str]]] = {}
    for row in table.data_rows():
        cell = table.cell_at(row, 0)
        text = ((cell.text_normalized if cell else "") or "").strip()
        if not text:
            continue
        hits = []
        # Every field in the title-block order is a possible anchor, whether or
        # not someone wrote an alias list for it. Iterating only over
        # field_aliases made 页码 and 版本 permanently unidentifiable, so a row
        # between them could never be bounded from below.
        candidates = dict.fromkeys(
            list(cfg.field_aliases) + list(cfg.title_block_field_order))
        for field_name in candidates:
            aliases = cfg.field_aliases.get(field_name) or [field_name]
            best = max(difflib.SequenceMatcher(None, alias, text).ratio()
                       for alias in aliases)
            if best >= cfg.alias_similarity_threshold:
                hits.append((best, field_name))
        if hits:
            scored[row] = sorted(hits, reverse=True)

    out: Dict[int, str] = {}
    for row, hits in scored.items():
        # A tie means the row could be either field. Taking the first would
        # file a value under a name nobody established.
        if len(hits) > 1 and abs(hits[0][0] - hits[1][0]) < 1e-6:
            continue
        out[row] = hits[0][1]

    # One field cannot occupy two rows. If it appears to, neither row is settled.
    counts = Counter(out.values())
    return {row: name for row, name in out.items() if counts[name] == 1}


# ---------------------------------------------------------------------------
# L1..L6 — the ladder
# ---------------------------------------------------------------------------

def _region(strategy: str, bbox, *, document_id, page_no, fields, anchors,
            reasons, width, height, alias_match=None,
            cfg: Optional[ReviewRegionConfig] = None) -> ResolvedReviewRegion:
    trust = STRATEGY_TRUST[strategy]
    ratio = area_ratio(bbox, width, height)
    # A "local" crop covering most of the page is a full-page call with a
    # friendlier name. Re-grade it rather than let it claim precision.
    if trust in (PRECISE, STRUCTURAL) and cfg and ratio > cfg.max_context_area_ratio:
        trust = CONTEXTUAL
        reasons = list(reasons) + [
            "re-graded to contextual: covers {:.0%} of the page".format(ratio)]
    return ResolvedReviewRegion(
        request_id="region:{}:p{}:{}".format(
            document_id, page_no, "+".join(sorted(fields))),
        document_id=document_id, page_no=page_no, target_fields=list(fields),
        bbox=list(bbox), strategy=strategy, trust_level=trust,
        source_anchors=list(anchors), area_ratio=ratio, reasons=list(reasons),
        localization_uncertain=(trust != PRECISE),
        answer_eligible=(trust in ANSWER_ELIGIBLE_TRUST),
        alias_match=alias_match)


def _unavailable(document_id, page_no, fields, reason_code, reasons):
    return ResolvedReviewRegion(
        request_id="region:{}:p{}:{}".format(
            document_id, page_no, "+".join(sorted(fields))),
        document_id=document_id, page_no=page_no, target_fields=list(fields),
        bbox=[], strategy=UNAVAILABLE, trust_level=DIAGNOSTIC,
        reasons=list(reasons), localization_uncertain=True,
        answer_eligible=False, unavailable_reason=reason_code)


def _big_enough(bbox, cfg: ReviewRegionConfig) -> bool:
    return (_valid(bbox) and bbox[2] - bbox[0] >= cfg.min_region_width
            and bbox[3] - bbox[1] >= cfg.min_region_height)


def resolve_field_region(
    field_name: str,
    *,
    document_id: str,
    page_no: int,
    width: int,
    height: int,
    cfg: ReviewRegionConfig,
    ocr_blocks: Sequence = (),
    table=None,
    key_value: bool = False,
    identified_rows: Optional[Dict[int, str]] = None,
    reasons: Sequence[str] = (),
) -> ResolvedReviewRegion:
    """L1 -> L4 for one field. L5/L6 are page-level and handled by the caller."""
    cells = list(table.cells) if table is not None else []
    identified_rows = identified_rows or {}
    trail: List[str] = list(reasons)

    # -- L1: the label is there, intact -----------------------------------
    exact = find_exact_label(field_name, ocr_blocks)
    if exact:
        text, label_bbox = exact
        bbox = pad(value_region_right_of(label_bbox, cfg, width, height, cells),
                   cfg, width, height)
        if _big_enough(bbox, cfg):
            return _region(EXACT_LABEL_RIGHT, bbox, document_id=document_id,
                           page_no=page_no, fields=[field_name],
                           anchors=["label:" + text],
                           reasons=trail + ["exact label text found"],
                           width=width, height=height, cfg=cfg)
        trail.append("exact label found but its value region was too small")

    # -- L2: a damaged label, matched conservatively -----------------------
    alias, alias_info = find_alias_label(field_name, ocr_blocks, cfg)
    if alias:
        text, label_bbox = alias
        bbox = pad(value_region_right_of(label_bbox, cfg, width, height, cells),
                   cfg, width, height)
        if _big_enough(bbox, cfg):
            return _region(ALIAS_LABEL_RIGHT, bbox, document_id=document_id,
                           page_no=page_no, fields=[field_name],
                           anchors=["alias:" + text],
                           reasons=trail + [
                               "alias match {!r} score={}".format(
                                   alias_info["matched_alias"],
                                   alias_info["score"])],
                           width=width, height=height,
                           alias_match=alias_info, cfg=cfg)
    elif alias_info and alias_info.get("ambiguous"):
        # Two candidates tied. WHICH field this is, is the open question;
        # taking the higher float would file a value under the wrong name.
        trail.append("alias match ambiguous: {}".format(alias_info["candidates"]))

    # -- L3: the row of a confirmed key/value table ------------------------
    if table is not None and key_value:
        row = next((r for r, name in identified_rows.items()
                    if name == field_name), None)
        strategy, anchors = TABLE_VALUE_CELL, []
        if row is None:
            row, anchors = infer_key_value_row(field_name, table, cfg,
                                               identified_rows)
            strategy = STRUCTURAL_ROW_INFERENCE
            if row is None:
                trail.extend(anchors)
                anchors = []
        else:
            anchors = ["row{}:left cell reads the field name".format(row)]
        if row is not None:
            cell = table.cell_at(row, 1)
            cell_bbox = list(getattr(cell, "bbox", []) or []) if cell else []
            if _valid(cell_bbox):
                bbox = pad(clip_to_image(cell_bbox, width, height),
                           cfg, width, height)
                if _big_enough(bbox, cfg):
                    return _region(strategy, bbox, document_id=document_id,
                                   page_no=page_no, fields=[field_name],
                                   anchors=anchors,
                                   reasons=trail + ["value cell of a confirmed "
                                                    "key/value table"],
                                   width=width, height=height, cfg=cfg)

    # -- L4: the table or title block it belongs to ------------------------
    table_bbox = list(getattr(table, "bbox", []) or []) if table is not None else []
    if _valid(table_bbox):
        bbox = pad(clip_to_image(table_bbox, width, height), cfg, width, height)
        if _big_enough(bbox, cfg):
            return _region(TABLE_OR_TITLE_BLOCK, bbox, document_id=document_id,
                           page_no=page_no, fields=[field_name],
                           anchors=["table:" + str(getattr(table, "table_id", ""))],
                           reasons=trail + ["field could not be located; the "
                                            "table containing it was"],
                           width=width, height=height, cfg=cfg)

    return _unavailable(document_id, page_no, [field_name], R_NO_STRUCTURE,
                        trail + ["no label, no table row and no table region"])


def full_page_diagnostic(
    *, document_id: str, page_no: int, width: int, height: int,
    fields: Sequence[str], cfg: ReviewRegionConfig, reasons: Sequence[str],
) -> ResolvedReviewRegion:
    """L5. Diagnosis only — never field evidence.

    One per page. It exists so a page that cannot be typed stops producing ZERO
    requests and vanishing from the queue; it does not mean the page is handled.
    """
    long_edge = max(width, height)
    scale = 1.0
    if long_edge > cfg.diagnostic_max_long_edge:
        scale = cfg.diagnostic_max_long_edge / long_edge
    if width * height * scale * scale > cfg.diagnostic_max_pixels:
        scale = min(scale, (cfg.diagnostic_max_pixels / (width * height)) ** 0.5)

    region = _region(FULL_PAGE_DIAGNOSTIC, [0.0, 0.0, float(width), float(height)],
                     document_id=document_id, page_no=page_no, fields=fields,
                     anchors=["whole page"], reasons=list(reasons),
                     width=width, height=height, cfg=None)
    region.scale = round(scale, 6)
    return region


def deduplicate(regions: Sequence[ResolvedReviewRegion],
                cfg: ReviewRegionConfig) -> List[ResolvedReviewRegion]:
    """Merge overlapping regions, keeping the most precise one.

    A page whose title block is cropped once per missing field pays for the
    same pixels five times. Merging keeps one region carrying every field and
    every reason, and the survivor is the tighter, more trusted crop — never
    the bigger one, which would quietly coarsen a precise localisation.
    """
    ordered = sorted(regions, key=lambda r: (TRUST_RANK[r.trust_level],
                                             r.area_ratio))
    kept: List[ResolvedReviewRegion] = []
    for region in ordered:
        if not region.available:
            kept.append(region)
            continue
        match = next(
            (k for k in kept if k.available and k.page_no == region.page_no
             and k.document_id == region.document_id
             and iou(k.bbox, region.bbox) >= cfg.dedup_iou_threshold), None)
        if match is None:
            kept.append(region)
            continue
        for name in region.target_fields:
            if name not in match.target_fields:
                match.target_fields.append(name)
        for reason in region.reasons:
            if reason not in match.reasons:
                match.reasons.append(reason)
        for anchor in region.source_anchors:
            if anchor not in match.source_anchors:
                match.source_anchors.append(anchor)
        match.request_id = "region:{}:p{}:{}".format(
            match.document_id, match.page_no,
            "+".join(sorted(match.target_fields)))
    return kept


def resolve_review_regions(
    *,
    document_id: str,
    page_no: int,
    image_size: Optional[Tuple[int, int]],
    ocr_result=None,
    parsed_table=None,
    completeness=None,
    drawing_type: Optional[str] = None,
    target_fields: Sequence[str] = (),
    config: ReviewRegionConfig,
    field_reasons: Optional[Dict[str, List[str]]] = None,
) -> List[ResolvedReviewRegion]:
    """The whole ladder for one page.

    `image_size` is (width, height); None means the page could not be read,
    which is the only condition under which nothing at all can be produced.
    """
    field_reasons = field_reasons or {}
    if image_size is None:
        return [_unavailable(document_id, page_no, list(target_fields) or ["*"],
                             R_IMAGE_MISSING, ["image missing or unreadable"])]
    width, height = image_size
    blocks = list(getattr(ocr_result, "blocks", []) or [])
    type_known = bool(drawing_type and drawing_type != "unknown")
    sparse = len(blocks) <= config.sparse_page_max_blocks

    page_level_reasons: List[str] = []
    if not type_known:
        page_level_reasons.append("drawing type unknown: no required-field list")
    if sparse:
        page_level_reasons.append(
            "OCR returned {} blocks (<= {})".format(
                len(blocks), config.sparse_page_max_blocks))
    if parsed_table is None:
        page_level_reasons.append("no table structure recovered")

    # A page with no type has no required-field list, so `target_fields` is
    # empty and every per-field path below is unreachable. Before this branch
    # existed, such a page produced ZERO requests and disappeared.
    if not target_fields:
        if not page_level_reasons:
            return []          # nothing wrong and nothing asked: a clean page
        if not config.enable_full_page_diagnostic:
            return [_unavailable(document_id, page_no, ["*"],
                                 R_DIAGNOSTIC_DISABLED,
                                 page_level_reasons + ["diagnostic disabled"])]
        return [full_page_diagnostic(
            document_id=document_id, page_no=page_no, width=width,
            height=height, fields=["*"], cfg=config,
            reasons=page_level_reasons)]

    key_value = False
    identified_rows: Dict[int, str] = {}
    if parsed_table is not None:
        from src.tables.cells import detect_key_value_layout
        key_value = detect_key_value_layout(parsed_table)
        if key_value:
            identified_rows = identify_key_value_rows(parsed_table, config)

    regions = [
        resolve_field_region(
            name, document_id=document_id, page_no=page_no, width=width,
            height=height, cfg=config, ocr_blocks=blocks, table=parsed_table,
            key_value=key_value, identified_rows=identified_rows,
            reasons=field_reasons.get(name, []))
        for name in target_fields
    ]

    # Every field failed, but the page itself may still be worth a look.
    if (all(not r.available for r in regions)
            and config.enable_full_page_diagnostic):
        regions.append(full_page_diagnostic(
            document_id=document_id, page_no=page_no, width=width, height=height,
            fields=list(target_fields), cfg=config,
            reasons=page_level_reasons + ["every per-field strategy failed"]))

    return deduplicate(regions, config)[:config.max_regions_per_page]
