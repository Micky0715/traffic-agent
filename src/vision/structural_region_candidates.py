"""Shadow experiment: can a damaged-label field be narrowed below a whole table?

The shipped resolver falls back to `table_or_title_block` for 8 of 11 targets —
a correct but coarse answer. This module explores whether grid structure,
identified neighbours and layout convention can propose FIELD-LEVEL candidates
instead. It is an experiment and is wired as one:

  * every candidate carries `promotion_status="shadow_only"`;
  * nothing here is imported by `resolve_review_regions`, the evidence policy,
    the executor or any decision path;
  * no candidate can change an execute / partial / abstain outcome;
  * no VLM is called.

Two rules shape the output. First, sub-scores are kept separately and never
collapsed into a single number on the record — a total of 0.6 tells a reader
nothing about whether the row was read, bounded or merely assumed. Second,
ambiguity is emitted rather than resolved: three plausible rows produce three
candidates and `ambiguity_count=3`, because picking one and calling it located
is exactly the dressed-up guess the grading system exists to prevent.

Forbidden inputs, by construction — none of these is a parameter: file names,
sample ids, gold bboxes, correct field values, VLM output, or any row observed
to be correct in an earlier report.
"""
from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "structural_shadow.yaml"

SHADOW_ONLY = "shadow_only"

KV_ROW = "kv_row_candidate"
SINGLE_SIDED = "single_sided_anchor"
ORDER_ONLY = "field_order_only"


@dataclass(frozen=True)
class ShadowConfig:
    enabled: bool = True
    promotion_status: str = SHADOW_ONLY
    threshold_provenance: str = "engineering_initial_value"
    calibrated_with_human_gold: bool = False
    weights: Dict[str, float] = field(default_factory=dict)
    single_sided_anchor_penalty: float = 0.55
    min_candidate_score: float = 0.05
    max_candidates_per_field: int = 6
    text_evidence_floor: float = 0.30
    field_order_provenance: str = "dataset_or_project_convention"
    field_order_by_drawing_type: Dict[str, List[str]] = field(default_factory=dict)

    def order_for(self, drawing_type: Optional[str]) -> List[str]:
        """Only this drawing type's ordering. Never another's.

        A fan wiring title block and a pump schedule are not laid out the same
        way; borrowing one list for the other would manufacture evidence.
        """
        return list(self.field_order_by_drawing_type.get(drawing_type or "", []))


def load_shadow_config(path: Path | None = None) -> ShadowConfig:
    raw = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    return ShadowConfig(**dict(raw.get("structural_shadow", {})))


@dataclass
class StructuralRegionCandidate:
    target_field: str
    bbox: List[float]
    strategy: str
    score: float
    evidence: Dict[str, float] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    ambiguity_count: int = 1
    promotion_status: str = SHADOW_ONLY
    row_index: Optional[int] = None
    anchors: List[str] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["bbox"] = [float(v) for v in self.bbox]
        payload["score"] = round(self.score, 4)
        payload["evidence"] = {k: round(v, 4) for k, v in self.evidence.items()}
        return payload


# ---------------------------------------------------------------------------
# sub-scores — each answers one question, and each is kept
# ---------------------------------------------------------------------------

def text_similarity_score(left_text: str, field_name: str,
                          aliases: Sequence[str], cfg: ShadowConfig) -> float:
    """How much the (possibly ruined) left cell still looks like this field."""
    if not left_text:
        return 0.0
    best = max(difflib.SequenceMatcher(None, alias, left_text).ratio()
               for alias in (aliases or [field_name]))
    return best if best >= cfg.text_evidence_floor else 0.0


def neighbor_anchor_score(row: int, identified: Dict[int, str],
                          order: Sequence[str], field_name: str) -> tuple:
    """Do identified rows above and below place this row correctly?

    Returns (score, assumptions, anchors). A one-sided bound is real evidence
    but weaker, and says so instead of being rounded up to the two-sided case.
    """
    if field_name not in order:
        return 0.0, ["field is not in this drawing type's order"], []
    index = order.index(field_name)
    before, after = set(order[:index]), set(order[index + 1:])

    above = [r for r, name in identified.items() if r < row and name in before]
    below = [r for r, name in identified.items() if r > row and name in after]
    anchors = ([f"above:row{max(above)}={identified[max(above)]}"] if above else []) + \
              ([f"below:row{min(below)}={identified[min(below)]}"] if below else [])

    if above and below:
        return 1.0, [], anchors
    if above or below:
        return 1.0, ["bounded on one side only; the row could lie further "
                     "in the unbounded direction"], anchors
    return 0.0, ["no identified neighbour places this row"], []


def field_order_score(row: int, rows: Sequence[int], order: Sequence[str],
                      field_name: str) -> float:
    """How close this row sits to where the convention expects the field.

    Weakest signal by design: the ordering is a project convention.
    """
    if field_name not in order or not rows:
        return 0.0
    expected = order.index(field_name) / max(1, len(order) - 1)
    actual = list(rows).index(row) / max(1, len(rows) - 1)
    return max(0.0, 1.0 - abs(expected - actual))


def grid_consistency_score(bbox: Sequence[float]) -> float:
    """The cell is a real, sanely-shaped region of the recovered grid."""
    if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return 0.0
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # A cell far taller than wide is usually a column artefact, not a value box.
    return 1.0 if width >= height else 0.5


def cell_content_score(text: str) -> float:
    """Does the right cell hold something that could be a value at all?

    An empty cell is not disqualifying — the value may simply be unreadable —
    but a cell with content is better evidence that the row carries one.
    """
    stripped = (text or "").strip()
    if not stripped:
        return 0.2
    return 1.0 if any(ch.isdigit() or ch.isalpha() for ch in stripped) else 0.5


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def generate_structural_candidates(
    parsed_table,
    ocr_blocks: Sequence,
    target_field: str,
    drawing_type: Optional[str],
    config: ShadowConfig,
    *,
    aliases: Optional[Sequence[str]] = None,
    identified_rows: Optional[Dict[int, str]] = None,
) -> List[StructuralRegionCandidate]:
    """Every row of a key/value table that could hold `target_field`.

    Returns a SET, not a winner. If three rows are plausible, three candidates
    come back with `ambiguity_count=3`; narrowing that to one here would
    reintroduce the confident guess this module was written to avoid.
    """
    if not config.enabled or parsed_table is None:
        return []

    from src.tables.cells import detect_key_value_layout
    if not detect_key_value_layout(parsed_table):
        return []

    identified_rows = identified_rows or {}
    order = config.order_for(drawing_type)
    aliases = list(aliases or [target_field])
    weights = config.weights
    rows = parsed_table.data_rows()

    # A row already read as a DIFFERENT field is not a candidate for this one.
    taken = {r for r, name in identified_rows.items() if name != target_field}

    candidates: List[StructuralRegionCandidate] = []
    for row in rows:
        if row in taken:
            continue
        value_cell = parsed_table.cell_at(row, 1)
        left_cell = parsed_table.cell_at(row, 0)
        bbox = list(getattr(value_cell, "bbox", []) or []) if value_cell else []
        if len(bbox) != 4:
            continue

        left_text = (getattr(left_cell, "text_normalized", "") or "") if left_cell else ""
        value_text = (getattr(value_cell, "text_normalized", "") or "")

        text = text_similarity_score(left_text, target_field, aliases, config)
        anchor, assumptions, anchor_names = neighbor_anchor_score(
            row, identified_rows, order, target_field)
        layout = field_order_score(row, rows, order, target_field)
        grid = grid_consistency_score(bbox)
        content = cell_content_score(value_text)

        evidence = {
            "text_similarity_score": text,
            "neighbor_anchor_score": anchor,
            "field_order_score": layout,
            "grid_consistency_score": grid,
            "cell_content_score": content,
        }
        total = (weights.get("text_similarity", 0) * text
                 + weights.get("neighbor_anchor", 0) * anchor
                 + weights.get("field_order", 0) * layout
                 + weights.get("grid_consistency", 0) * grid
                 + weights.get("cell_content", 0) * content)

        strategy = KV_ROW
        one_sided = any("one side only" in a for a in assumptions)
        if one_sided:
            total *= config.single_sided_anchor_penalty
            strategy = SINGLE_SIDED
        elif anchor == 0.0 and text == 0.0:
            strategy = ORDER_ONLY
            assumptions = assumptions + [
                "placed by layout convention alone; no text and no anchor"]

        if total < config.min_candidate_score:
            continue

        candidates.append(StructuralRegionCandidate(
            target_field=target_field, bbox=bbox, strategy=strategy,
            score=total, evidence=evidence, assumptions=assumptions,
            row_index=row, anchors=anchor_names,
            promotion_status=config.promotion_status,
            provenance={
                "field_order": config.field_order_provenance,
                "drawing_type": drawing_type or "unknown",
                "calibrated_with_human_gold": str(config.calibrated_with_human_gold),
            }))

    candidates.sort(key=lambda c: -c.score)
    candidates = candidates[:config.max_candidates_per_field]
    for candidate in candidates:
        candidate.ambiguity_count = len(candidates)
    return candidates


def score_gap(candidates: Sequence[StructuralRegionCandidate]) -> Optional[float]:
    """How far the top candidate leads. A small gap IS the finding."""
    if len(candidates) < 2:
        return None
    return round(candidates[0].score - candidates[1].score, 4)
