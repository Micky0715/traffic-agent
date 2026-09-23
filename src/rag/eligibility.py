"""Whether a retrieved chunk may be used as evidence.

This layer exists because BM25 always returns its top K. Returning three
results is not the same as having three pieces of evidence, and conflating the
two is how a system answers a question nobody's documents contain: the
retriever did its job, the chunks are simply about something else.

So `retrieved_candidates` and `eligible_evidence` are counted separately
everywhere, and only the second may support an answer. When BM25 returns three
rows and none of them covers the requested field for the requested device, the
correct outcome is abstain — with the three candidates still reported, so the
refusal is inspectable.

This is also the right place to stop the "irrelevant query still returns
results" problem. The root cause is that CJK tokenizes to single characters and
an unrelated question shares function words like 的 with almost any text. A
score threshold tuned to make one query behave is guesswork; requiring that a
chunk actually be about the right entity and the right field is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.rag.query import QueryPlan
from src.vision.schemas import DocumentChunk

# Which chunk kinds can answer which task. A clause cannot answer a drawing
# field query no matter how well it scores.
TASK_COMPATIBLE_TYPES: Dict[str, set] = {
    "drawing_field_query": {"drawing_field", "table_row", "drawing_metadata"},
    "equipment_metric_query": {"drawing_field", "table_row"},
    "regulation_lookup": {"clause", "section"},
}


@dataclass
class EligibilityResult:
    eligible: bool
    reasons: List[str] = field(default_factory=list)
    matched_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {"eligible": self.eligible, "reasons": self.reasons,
                "matched_fields": self.matched_fields}


def _entity_matches(plan: QueryPlan, chunk: DocumentChunk,
                    resolved_entity: Optional[str],
                    resolved_documents: Optional[Sequence[str]]) -> bool:
    """Does this chunk belong to the entity the query named?

    Two ways to belong, and the second is not optional. A chunk can carry an
    entity_id of its own — a row in a multi-device table does. But on a
    single-device drawing the fields are PAGE-LEVEL and carry no entity_id at
    all; they belong to the device because the drawing does. Matching on
    entity_id alone rejects every one of them, which is why the entity is
    resolved to documents first.
    """
    wanted = resolved_entity or plan.asset_id or plan.drawing_no
    if not wanted:
        return True
    if chunk.entity_id:
        return wanted in chunk.entity_id or chunk.entity_id in wanted
    if resolved_documents and chunk.document_id in set(resolved_documents):
        return True
    drawing_no = (chunk.metadata or {}).get("drawing_no") or ""
    if drawing_no and (wanted in drawing_no or drawing_no in wanted):
        return True
    return bool(chunk.field_value and wanted in chunk.field_value)


def evaluate_candidate_eligibility(
    query_plan: QueryPlan,
    chunk: DocumentChunk,
    *,
    resolved_entity: Optional[str] = None,
    resolved_documents: Optional[Sequence[str]] = None,
) -> EligibilityResult:
    """All seven conditions must hold. Each failure is named."""
    reasons: List[str] = []

    # 1. traceable to a document and a page
    if not chunk.document_id or chunk.page_start is None:
        reasons.append("NOT_TRACEABLE")

    # 2. right entity
    if not _entity_matches(query_plan, chunk, resolved_entity, resolved_documents):
        reasons.append("ENTITY_MISMATCH")

    # 3. covers a requested field
    matched = []
    if query_plan.required_fields:
        if chunk.field_name and chunk.field_name in query_plan.required_fields:
            matched.append(chunk.field_name)
        elif chunk.header_paths:
            for path in chunk.header_paths:
                leaf = path[-1] if path else ""
                if leaf in query_plan.required_fields:
                    matched.append(leaf)
        if not matched:
            reasons.append("FIELD_NOT_COVERED")
    else:
        # No target field named: the query cannot be answered from evidence
        # yet, and the caller is expected to ask rather than pick a field.
        reasons.append("NO_TARGET_FIELD_IN_QUERY")

    # 4. non-empty value
    if chunk.field_name and not (chunk.field_value or "").strip():
        reasons.append("EMPTY_FIELD_VALUE")

    # 5. not judged invalid
    if (chunk.metadata or {}).get("validation_status") == "invalid":
        reasons.append("FIELD_VALUE_INVALID")

    # 6. no unresolved conflict on this chunk
    if (chunk.metadata or {}).get("conflict"):
        reasons.append("UNRESOLVED_CONFLICT")

    # 7. chunk kind can answer this task
    compatible = TASK_COMPATIBLE_TYPES.get(query_plan.task_type)
    if compatible is not None and chunk.content_type not in compatible:
        reasons.append("CHUNK_TYPE_INCOMPATIBLE")

    return EligibilityResult(eligible=not reasons, reasons=reasons,
                             matched_fields=sorted(set(matched)))


def partition_candidates(
    query_plan: QueryPlan,
    candidates: Sequence,
    *,
    resolved_entity: Optional[str] = None,
    resolved_documents: Optional[Sequence[str]] = None,
) -> Dict[str, list]:
    """Split retrieved candidates into eligible evidence and rejected ones.

    Rejected candidates are kept, not dropped: a refusal that cannot show what
    it looked at and why it turned each thing down is indistinguishable from a
    failure to look.
    """
    eligible, rejected = [], []
    for candidate in candidates:
        chunk = getattr(candidate, "chunk", candidate)
        result = evaluate_candidate_eligibility(
            query_plan, chunk, resolved_entity=resolved_entity,
            resolved_documents=resolved_documents)
        record = {"candidate": candidate, "chunk": chunk, "eligibility": result}
        (eligible if result.eligible else rejected).append(record)
    return {"eligible": eligible, "rejected": rejected}
