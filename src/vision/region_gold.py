"""Human-reviewed bbox gold: the data contract, and what it refuses.

This repo has reported crop availability for two rounds and has never reported
localisation accuracy, because accuracy needs an answer key and the only thing
available was the system's own output. A gold set built from predictions
measures whether the code agrees with itself.

So the type below is written to make that mistake loud rather than possible:

  * `label_status` starts at `unreviewed` and nothing in this file can move it
    to `human_reviewed` — only a person editing the record can, and doing so
    without an annotator name is rejected;
  * `gold_bbox` has no default and no derivation. There is deliberately no
    constructor that takes a ResolvedReviewRegion;
  * a record whose image hash no longer matches the file is refused at
    evaluation time, not silently re-used against different pixels.

One record is one (page, field). A gold that says only "this drawing" cannot
answer whether 控制柜编号 was found in the right place.

Schema v2 (`region_gold/2`) separates three questions v1 could not tell apart:

  value_visible       is there a value printed on the page at all?
  value_legible       can a person reliably read its characters?
  association_status  can a person confirm it belongs to THIS field?
                      confirmed | ambiguous | unassigned | unknown

v1 had no way to say "I can read FAN-CAB-24, but its label is destroyed so I
cannot tell whether it is the 控制柜编号". Its only honest option was to mark
the value invisible, which is false. v1 even forbade the combination
field_visible=false + value_visible=true — the exact shape of that case.

Compatibility rules, none of which infer anything:
  * a record with no `schema_version` is v1, and its `value_visible` keeps the
    meaning it was annotated under (the v1 guide used it for legibility); it is
    never re-read as v2 "presence" and never used to fill `value_legible`;
  * a missing `association_status` is `unknown`, never `confirmed`;
  * only a person sets `confirmed`. Nothing here reads OCR or resolver output.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

UNREVIEWED = "unreviewed"
HUMAN_REVIEWED = "human_reviewed"
EXCLUDED = "excluded"
LABEL_STATUSES = (UNREVIEWED, HUMAN_REVIEWED, EXCLUDED)

VALUE_CELL = "value_cell"
LABEL_VALUE_PAIR = "label_value_pair"
TABLE_REGION = "table_region"
TITLE_BLOCK = "title_block"
FULL_PAGE = "full_page"
UNLOCATABLE = "unlocatable"
REGION_TYPES = (VALUE_CELL, LABEL_VALUE_PAIR, TABLE_REGION, TITLE_BLOCK,
                FULL_PAGE, UNLOCATABLE)

# Region types that describe a located field. A gold record of TABLE_REGION or
# FULL_PAGE is a true statement about where the field lives, but it is not a
# field-level answer key and must not be scored as one.
FIELD_LEVEL_TYPES = (VALUE_CELL, LABEL_VALUE_PAIR)

SCHEMA_V1 = "region_gold/1"
SCHEMA_V2 = "region_gold/2"
SCHEMA_VERSIONS = (SCHEMA_V1, SCHEMA_V2)

CONFIRMED = "confirmed"      # a person confirmed the value belongs to this field
AMBIGUOUS = "ambiguous"      # value readable; could belong to several fields
UNASSIGNED = "unassigned"    # value readable; cannot be assigned to this field
UNKNOWN = "unknown"          # not annotated (all v1 data), or value unreadable
ASSOCIATION_STATUSES = (CONFIRMED, AMBIGUOUS, UNASSIGNED, UNKNOWN)
# A candidate value is on the page but its field is not settled.
ATTRIBUTION_OPEN = (AMBIGUOUS, UNASSIGNED)

# Present only on v2 records. A v1 record carrying any of them is a
# half-migrated row and is rejected rather than guessed at.
V2_ONLY_FIELDS = ("value_legible", "candidate_bbox", "candidate_fields",
                  "migrated_from")


class GoldContractError(ValueError):
    """A record that cannot be trusted as an answer key."""


@dataclass
class RegionGoldRecord:
    record_id: str
    document_id: str
    page_no: int
    image_path: str
    image_sha256: str
    target_field: str
    label_status: str = UNREVIEWED
    gold_bbox: Optional[List[float]] = None
    gold_region_type: Optional[str] = None
    field_visible: Optional[bool] = None
    value_visible: Optional[bool] = None
    needs_visual_review: bool = True
    unlocatable_reason: Optional[str] = None
    annotator: Optional[str] = None
    annotated_at: Optional[str] = None
    notes: Optional[str] = None
    # ---- schema v2; every one optional so v1 files still load --------------
    # Default v1: a record whose version was never written down is judged by
    # the rules it was annotated under, not promoted to the newer ones.
    schema_version: str = SCHEMA_V1
    value_legible: Optional[bool] = None
    association_status: str = UNKNOWN
    # Where a readable but UNATTRIBUTED value sits. Kept apart from gold_bbox,
    # which only ever means "a person confirmed this field is here".
    candidate_bbox: Optional[List[float]] = None
    candidate_fields: Optional[List[str]] = None
    # Set by the migration tool on v1 rows it restamped as v2. Such a row keeps
    # v1 tolerances: its value_visible has v1 semantics and its attribution
    # was never asked.
    migrated_from: Optional[str] = None

    # ---- construction ----------------------------------------------------

    @classmethod
    def pending(cls, *, document_id: str, page_no: int, image_path: Path,
                target_field: str, notes: Optional[str] = None) -> "RegionGoldRecord":
        """A record awaiting a person. Every conclusion field stays empty.

        There is no `from_prediction` classmethod and there will not be one:
        the point of this file is that a gold bbox has exactly one source.
        """
        return cls(
            record_id=f"{document_id}:p{page_no}:{target_field}",
            document_id=document_id, page_no=page_no,
            image_path=str(image_path).replace("\\", "/"),
            image_sha256=sha256_file(image_path),
            target_field=target_field,
            label_status=UNREVIEWED, gold_bbox=None, gold_region_type=None,
            field_visible=None, value_visible=None, needs_visual_review=True,
            annotator=None, annotated_at=None, notes=notes,
            schema_version=SCHEMA_V2, value_legible=None,
            association_status=UNKNOWN, candidate_bbox=None,
            candidate_fields=None, migrated_from=None)

    # ---- queries ---------------------------------------------------------

    @property
    def is_reviewed(self) -> bool:
        return self.label_status == HUMAN_REVIEWED

    @property
    def counts_toward_metrics(self) -> bool:
        """`unreviewed` and `excluded` are both outside the denominator.

        Counting an unreviewed record as a miss would turn "nobody has looked
        at this yet" into evidence that the system failed.
        """
        return self.label_status == HUMAN_REVIEWED

    @property
    def is_field_level(self) -> bool:
        return self.gold_region_type in FIELD_LEVEL_TYPES

    @property
    def effective_association(self) -> str:
        """v1 never recorded attribution, so it is `unknown` — whatever the
        bbox or the notes suggest. Reading 'value cell containing FAN-A23-01'
        in a note as confirmation is the inference this schema forbids."""
        if self.schema_version == SCHEMA_V1:
            return UNKNOWN
        return self.association_status

    @property
    def is_native_v2(self) -> bool:
        return self.schema_version == SCHEMA_V2 and self.migrated_from is None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate_record(record: RegionGoldRecord,
                    image_size: Optional[tuple] = None) -> List[str]:
    """Every way one record can fail to be an answer key. Returns problems."""
    problems: List[str] = []

    if record.label_status not in LABEL_STATUSES:
        problems.append(f"label_status {record.label_status!r} is not one of "
                        f"{LABEL_STATUSES}")
    if record.gold_region_type is not None and \
            record.gold_region_type not in REGION_TYPES:
        problems.append(f"gold_region_type {record.gold_region_type!r} is not "
                        f"one of {REGION_TYPES}")

    problems.extend(_schema_version_problems(record))

    if record.label_status == UNREVIEWED:
        # A pending record carrying conclusions is the failure this whole
        # module exists to prevent.
        for name in ("gold_bbox", "gold_region_type", "annotator", "annotated_at",
                     "value_legible", "candidate_bbox", "candidate_fields"):
            if getattr(record, name) is not None:
                problems.append(
                    f"unreviewed record carries {name}; a pending record must "
                    f"hold no conclusion")
        if record.association_status != UNKNOWN:
            problems.append(
                "unreviewed record carries association_status="
                f"{record.association_status!r}; a pending record must hold "
                "no conclusion")
        return problems

    if record.label_status == EXCLUDED:
        if not record.notes:
            problems.append("excluded record must say why in notes")
        return problems

    # -- human_reviewed from here -----------------------------------------
    if not record.annotator:
        problems.append("human_reviewed record has no annotator")
    if not record.annotated_at:
        problems.append("human_reviewed record has no annotated_at")
    if record.gold_region_type is None:
        problems.append("human_reviewed record has no gold_region_type")

    if record.schema_version == SCHEMA_V2:
        problems.extend(_v2_attribution_problems(record, image_size))

    if record.gold_region_type == UNLOCATABLE:
        if not record.unlocatable_reason:
            problems.append("unlocatable record has no unlocatable_reason")
        if record.gold_bbox:
            problems.append("unlocatable record must not carry a bbox")
        return problems

    bbox = record.gold_bbox
    if not bbox:
        problems.append("human_reviewed locatable record has no gold_bbox")
        return problems
    if len(bbox) != 4:
        problems.append(f"gold_bbox must be [x0,y0,x1,y1], got {len(bbox)} values")
        return problems
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        problems.append(f"gold_bbox has zero or negative area: {bbox}")
    if min(x0, y0) < 0:
        problems.append(f"gold_bbox has negative coordinates: {bbox}")
    if image_size:
        width, height = image_size
        if x1 > width or y1 > height:
            problems.append(
                f"gold_bbox {bbox} extends past the image ({width}x{height})")

    # visibility vs bbox
    if record.field_visible is False and bbox:
        problems.append("field_visible=false but a gold_bbox was given")
    # v1 rule, kept for v1: it assumed a value cannot be seen without its
    # field. In v2 that exact combination is the ambiguous/unassigned case,
    # whose own constraints are checked in _v2_attribution_problems.
    if record.value_visible is True and record.field_visible is False \
            and record.effective_association not in ATTRIBUTION_OPEN:
        problems.append("value_visible=true while field_visible=false")
    return problems


def _valid_box(box) -> bool:
    return (isinstance(box, list) and len(box) == 4
            and box[2] > box[0] and box[3] > box[1]
            and min(box[0], box[1]) >= 0)


def _schema_version_problems(record: RegionGoldRecord) -> List[str]:
    problems: List[str] = []
    if record.schema_version not in SCHEMA_VERSIONS:
        problems.append(f"schema_version {record.schema_version!r} is not one of "
                        f"{SCHEMA_VERSIONS}")
        return problems
    if record.association_status not in ASSOCIATION_STATUSES:
        problems.append(f"association_status {record.association_status!r} is "
                        f"not one of {ASSOCIATION_STATUSES}")
    if record.schema_version == SCHEMA_V1:
        # Half-migrated: v2 fields on a v1 row. Guessing which version was
        # meant would silently choose which rules apply.
        carried = [n for n in V2_ONLY_FIELDS if getattr(record, n) is not None]
        if record.association_status != UNKNOWN:
            carried.append("association_status")
        if carried:
            problems.append(
                f"v1 record carries v2 fields {carried}; set schema_version "
                f"to {SCHEMA_V2!r} or remove them")
    if record.migrated_from is not None and record.migrated_from != SCHEMA_V1:
        problems.append(f"migrated_from {record.migrated_from!r} is not "
                        f"{SCHEMA_V1!r}")
    return problems


def _v2_attribution_problems(record: RegionGoldRecord,
                             image_size: Optional[tuple]) -> List[str]:
    """The three questions, and the combinations that contradict each other."""
    problems: List[str] = []
    status = record.association_status
    visible, legible = record.value_visible, record.value_legible
    migrated = record.migrated_from is not None

    if visible is False and legible is True:
        problems.append("value_legible=true while value_visible=false")

    if status == CONFIRMED:
        if not _valid_box(record.gold_bbox) or record.gold_region_type == UNLOCATABLE:
            problems.append("association_status=confirmed needs a valid gold_bbox "
                            "and a locatable gold_region_type")
        if legible is not True:
            # An unreadable value cannot be confirmed as anything.
            problems.append("association_status=confirmed needs value_legible=true")
        if record.candidate_bbox is not None:
            problems.append("confirmed record must not carry candidate_bbox; "
                            "the confirmed location is gold_bbox")

    if status in ATTRIBUTION_OPEN:
        if visible is not True or legible is not True:
            problems.append(f"association_status={status} needs value_visible=true "
                            "and value_legible=true; an unreadable value is "
                            "`unknown`, not unattributed")
        if record.gold_bbox is not None:
            problems.append(f"association_status={status} must not carry "
                            "gold_bbox; put the value's position in candidate_bbox")
        if record.gold_region_type != UNLOCATABLE:
            problems.append(f"association_status={status} needs "
                            "gold_region_type=unlocatable: the FIELD cannot be "
                            "located, even though a candidate value can")
        if record.candidate_bbox is not None:
            if not _valid_box(record.candidate_bbox):
                problems.append("candidate_bbox is not a valid xyxy box: "
                                f"{record.candidate_bbox}")
            elif image_size and (record.candidate_bbox[2] > image_size[0]
                                 or record.candidate_bbox[3] > image_size[1]):
                problems.append(f"candidate_bbox {record.candidate_bbox} extends "
                                f"past the image {image_size}")
    elif record.candidate_bbox is not None and status != CONFIRMED:
        problems.append("candidate_bbox is only meaningful for ambiguous or "
                        "unassigned attribution")

    if status == AMBIGUOUS:
        fields = record.candidate_fields or []
        if len(set(fields)) < 2 or record.target_field not in fields:
            problems.append("association_status=ambiguous needs candidate_fields "
                            "listing at least two fields, including target_field")
    elif record.candidate_fields is not None and status != UNASSIGNED:
        problems.append("candidate_fields is only meaningful for ambiguous or "
                        "unassigned attribution")

    if legible is False and status != UNKNOWN:
        problems.append("value_legible=false allows only association_status=unknown")

    if not migrated:
        # Native v2 annotations are held to the full contract. A migrated v1
        # row is not: its attribution was simply never asked.
        if status == UNKNOWN and legible is True:
            problems.append("value is legible but association_status is unknown; "
                            "decide confirmed, ambiguous or unassigned")
        if status == UNKNOWN and record.gold_bbox is not None:
            problems.append("native v2 gold_bbox means a person located THIS "
                            "field; set association_status=confirmed, or move "
                            "the box to candidate_bbox")
        if visible is True and legible is None:
            problems.append("native v2 record with value_visible=true must set "
                            "value_legible")
    return problems


def image_hash_matches(record: RegionGoldRecord, root: Path) -> bool:
    """A gold bbox belongs to the pixels it was drawn on. If the file changed,
    the record describes a picture that no longer exists."""
    path = root / record.image_path
    return path.exists() and sha256_file(path) == record.image_sha256


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def write_jsonl(records: Iterable[RegionGoldRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    path.write_text("\n".join(r.to_json() for r in rows) + "\n", encoding="utf-8")
    return len(rows)


def read_jsonl(path: Path) -> List[RegionGoldRecord]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(RegionGoldRecord(**json.loads(line)))
    return out


def reviewed_only(records: Iterable[RegionGoldRecord]) -> List[RegionGoldRecord]:
    return [r for r in records if r.counts_toward_metrics]


def status_summary(records: Iterable[RegionGoldRecord]) -> Dict[str, int]:
    summary = {status: 0 for status in LABEL_STATUSES}
    for record in records:
        summary[record.label_status] = summary.get(record.label_status, 0) + 1
    return summary
