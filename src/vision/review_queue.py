"""A local, in-process queue of regions a person should look at.

A list in a file. Not a message broker, not an async service, and it must not
be described as one — there is no worker, no delivery guarantee and no retry.

Ordering puts the silent failures first. A page OCR returned at 0.996
confidence with one sixth of its fields present is the case that reaches a
dispatcher as a confident wrong answer; an ambiguous cell, which this repo
measured as a false positive 57 times over, goes last.

Idempotent on (image hash, bbox, target fields, prompt version, schema
version). Re-running the builder after nothing changed must not grow the queue.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# Lower number is looked at first.
PRIORITY_SILENT_MISS = 10           # high OCR confidence, low completeness
PRIORITY_INVALID_VALUE = 20
PRIORITY_UNRESOLVED_CONFLICT = 30
PRIORITY_PRECISE_CROP = 40
PRIORITY_STRUCTURAL_CROP = 50
PRIORITY_CONTEXTUAL_CROP = 60
PRIORITY_DIAGNOSTIC = 70
PRIORITY_CELL_SPANS_COLUMNS = 99    # measured false positive; last, always

TRUST_PRIORITY = {
    "precise": PRIORITY_PRECISE_CROP,
    "structural": PRIORITY_STRUCTURAL_CROP,
    "contextual": PRIORITY_CONTEXTUAL_CROP,
    "diagnostic": PRIORITY_DIAGNOSTIC,
}

PENDING = "pending"
DONE = "done"
SKIPPED = "skipped"


@dataclass
class ReviewQueueItem:
    item_id: str
    priority: int
    document_id: str
    page_no: int
    target_fields: List[str]
    strategy: str
    trust_level: str
    bbox: List[float]
    reason_codes: List[str] = field(default_factory=list)
    estimated_area_ratio: float = 0.0
    review_status: str = PENDING
    image_sha256: str = ""
    prompt_version: str = ""
    schema_version: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def dedup_key(*, image_sha256: str, bbox: Sequence[float],
              target_fields: Sequence[str], prompt_version: str,
              schema_version: str) -> str:
    """Identity of a unit of review work.

    Rounded to the pixel: two crops differing by a float epsilon are the same
    question, and letting them differ would let the queue grow on every rebuild.
    """
    payload = "|".join([
        image_sha256,
        ",".join(f"{round(float(v))}" for v in bbox),
        ",".join(sorted(target_fields)),
        prompt_version, schema_version,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def priority_for(*, trust_level: str, reason_codes: Sequence[str],
                 ocr_confidence: Optional[float] = None,
                 completeness: Optional[float] = None,
                 silent_miss_confidence: float = 0.90,
                 silent_miss_completeness: float = 0.50) -> int:
    """Most urgent applicable reason wins."""
    codes = set(reason_codes)
    if (ocr_confidence is not None and completeness is not None
            and ocr_confidence >= silent_miss_confidence
            and completeness <= silent_miss_completeness):
        return PRIORITY_SILENT_MISS
    if "invalid_field_value" in codes:
        return PRIORITY_INVALID_VALUE
    if "ocr_vlm_conflict" in codes:
        return PRIORITY_UNRESOLVED_CONFLICT
    if codes == {"cell_assignment_uncertain"}:
        # Only when it is the ONLY reason. Alongside a real one it does not
        # drag the item to the back.
        return PRIORITY_CELL_SPANS_COLUMNS
    return TRUST_PRIORITY.get(trust_level, PRIORITY_DIAGNOSTIC)


class ReviewQueue:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._items: Dict[str, ReviewQueueItem] = {}
        if self.path and self.path.exists():
            for row in json.loads(self.path.read_text(encoding="utf-8")):
                item = ReviewQueueItem(**row)
                self._items[item.item_id] = item

    def __len__(self) -> int:
        return len(self._items)

    def add(self, *, document_id: str, page_no: int, target_fields: Sequence[str],
            strategy: str, trust_level: str, bbox: Sequence[float],
            reason_codes: Sequence[str], image_sha256: str,
            prompt_version: str, schema_version: str,
            estimated_area_ratio: float = 0.0,
            ocr_confidence: Optional[float] = None,
            completeness: Optional[float] = None) -> bool:
        """Returns True if this was new. An existing item is left alone —
        including its review_status, so a rebuild never resets someone's work."""
        item_id = dedup_key(image_sha256=image_sha256, bbox=bbox,
                            target_fields=target_fields,
                            prompt_version=prompt_version,
                            schema_version=schema_version)
        if item_id in self._items:
            return False
        self._items[item_id] = ReviewQueueItem(
            item_id=item_id,
            priority=priority_for(trust_level=trust_level,
                                  reason_codes=reason_codes,
                                  ocr_confidence=ocr_confidence,
                                  completeness=completeness),
            document_id=document_id, page_no=page_no,
            target_fields=sorted(target_fields), strategy=strategy,
            trust_level=trust_level, bbox=[float(v) for v in bbox],
            reason_codes=sorted(reason_codes),
            estimated_area_ratio=round(float(estimated_area_ratio), 6),
            image_sha256=image_sha256, prompt_version=prompt_version,
            schema_version=schema_version)
        return True

    def pending(self) -> List[ReviewQueueItem]:
        return sorted((i for i in self._items.values()
                       if i.review_status == PENDING),
                      key=lambda i: (i.priority, i.document_id,
                                     ",".join(i.target_fields)))

    def all_items(self) -> List[ReviewQueueItem]:
        return sorted(self._items.values(),
                      key=lambda i: (i.priority, i.document_id))

    def mark(self, item_id: str, status: str) -> None:
        if status not in (PENDING, DONE, SKIPPED):
            raise ValueError(f"unknown review_status {status!r}")
        self._items[item_id].review_status = status

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([i.to_dict() for i in self.all_items()],
                       ensure_ascii=False, indent=2), encoding="utf-8")

    def summary(self) -> Dict[str, object]:
        return {
            "items": len(self._items),
            "pending": sum(1 for i in self._items.values()
                           if i.review_status == PENDING),
            "by_priority": {
                str(p): sum(1 for i in self._items.values() if i.priority == p)
                for p in sorted({i.priority for i in self._items.values()})},
            "by_trust_level": {
                t: sum(1 for i in self._items.values() if i.trust_level == t)
                for t in sorted({i.trust_level for i in self._items.values()})},
            "note": ("a local list, not a message broker: no worker, no "
                     "delivery guarantee, no retry"),
        }
