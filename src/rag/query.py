"""Turn a user question into a retrieval plan.

Reuses the existing entity extraction (src.extractors) rather than inventing a
second one — two extractors would drift and then disagree about what the query
even asked for.

No LLM is called and no missing field is filled in by guessing. If the question
does not name a target field, that is recorded as "not specified" and the
caller decides whether to ask; inventing a plausible field would produce a
confident answer to a question nobody asked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.extractors import extract_slots, find_code_mentions
from src.normalize import normalize_text
from src.rag.config import QueryParsingConfig


@dataclass
class FieldRequest:
    """One field the query asked for, and how that was decided.

    match_method is recorded because a synonym hit and a literal hit carry
    different confidence, and a report that cannot tell them apart cannot say
    whether the synonym table is doing any work.
    """

    surface: str                # what the user wrote
    field_name: Optional[str]   # canonical field, or None when unmatched
    match_method: str           # exact | alias | normalized | unmatched


@dataclass
class QueryPlan:
    raw_query: str
    normalized_query: str
    task_type: str
    asset_id: Optional[str] = None
    asset_candidates: List[str] = field(default_factory=list)
    drawing_no: Optional[str] = None
    regulation_code: Optional[str] = None
    required_fields: List[str] = field(default_factory=list)
    field_requests: List[FieldRequest] = field(default_factory=list)
    unmatched_field_terms: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def has_entity(self) -> bool:
        return bool(self.asset_id or self.drawing_no)

    @property
    def entity_ambiguous(self) -> bool:
        """Several equipment ids and no way to tell which one is meant.

        Distinct from "no entity at all": both need the user, but the question
        put to them is different."""
        return len(self.asset_candidates) > 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "task_type": self.task_type,
            "asset_id": self.asset_id,
            "asset_candidates": self.asset_candidates,
            "drawing_no": self.drawing_no,
            "regulation_code": self.regulation_code,
            "required_fields": self.required_fields,
            "field_requests": [
                {"surface": r.surface, "field_name": r.field_name,
                 "match_method": r.match_method} for r in self.field_requests],
            "unmatched_field_terms": self.unmatched_field_terms,
            "notes": self.notes,
        }


def _match_field(term: str, cfg: QueryParsingConfig) -> FieldRequest:
    if term in cfg.canonical_fields:
        return FieldRequest(surface=term, field_name=term, match_method="exact")
    for canonical, aliases in cfg.field_synonyms.items():
        if term in aliases:
            return FieldRequest(surface=term, field_name=canonical, match_method="alias")
    stripped = term.replace(" ", "")
    if stripped in cfg.canonical_fields:
        return FieldRequest(surface=term, field_name=stripped, match_method="normalized")
    for canonical, aliases in cfg.field_synonyms.items():
        if stripped in {a.replace(" ", "") for a in aliases}:
            return FieldRequest(surface=term, field_name=canonical,
                                match_method="normalized")
    return FieldRequest(surface=term, field_name=None, match_method="unmatched")


def parse_query(query: str, cfg: QueryParsingConfig,
                task_type: str = "drawing_field_query") -> QueryPlan:
    """Extract entity and target fields. Nothing is inferred that was not said."""
    normalized = normalize_text(query)
    slots = extract_slots(normalized)
    mentions = find_code_mentions(normalized)

    raw_asset = slots.get("asset_id")
    candidates = [str(x) for x in (raw_asset if isinstance(raw_asset, list)
                                   else [raw_asset] if raw_asset else [])]
    asset_id = candidates[0] if len(candidates) == 1 else None

    plan = QueryPlan(
        raw_query=query,
        normalized_query=normalized,
        task_type=task_type,
        asset_id=asset_id,
        asset_candidates=candidates,
        drawing_no=slots.get("drawing_no"),
        regulation_code=slots.get("regulation_code"),
    )

    # Longest surface first: "控制柜编号" must win over the substring "编号".
    seen_spans: List[tuple[int, int]] = []
    surfaces = sorted(cfg.all_field_surfaces(), key=len, reverse=True)
    for surface in surfaces:
        start = normalized.find(surface)
        if start < 0:
            continue
        end = start + len(surface)
        if any(start < s_end and s_start < end for s_start, s_end in seen_spans):
            continue
        seen_spans.append((start, end))
        request = _match_field(surface, cfg)
        plan.field_requests.append(request)
        if request.field_name and request.field_name not in plan.required_fields:
            plan.required_fields.append(request.field_name)

    if not plan.required_fields:
        plan.notes.append("no_target_field_specified")
    if plan.entity_ambiguous:
        plan.notes.append("multiple_entity_candidates")
    if not plan.has_entity and not candidates:
        plan.notes.append("no_entity_in_query")
    if mentions:
        plan.notes.append(f"code_mentions:{len(mentions)}")
    return plan


def resolve_entity(plan: QueryPlan, index) -> Dict[str, object]:
    """Resolve the named entity to the documents that contain it.

    Three outcomes, deliberately distinct because they deserve different
    answers: the query named nothing, the query named something this corpus
    does not have, and the query named something we can point at.
    """
    if isinstance(index, dict):
        entity_index: Dict[str, set] = index
    else:                       # a bare set of names, no document mapping
        entity_index = {name: set() for name in index if name}

    named = plan.asset_id or plan.drawing_no
    if not named:
        return {"status": "absent_from_query", "entity": None, "document_ids": []}
    if named in entity_index:
        return {"status": "present", "entity": named,
                "document_ids": sorted(entity_index[named])}

    # An equipment id often appears inside a longer name on the page
    # ("A13风机" inside "A13风机接线图"); a containment match is a resolution,
    # not a guess. Several documents may match, and all of them are returned —
    # narrowing to one arbitrarily would answer about the wrong drawing.
    documents: set = set()
    matched: List[str] = []
    for candidate, docs in entity_index.items():
        if named in candidate or candidate in named:
            documents |= docs
            matched.append(candidate)
    if documents:
        return {"status": "present", "entity": sorted(matched)[0],
                "matched_by": "partial", "queried": named,
                "matched_names": sorted(matched),
                "document_ids": sorted(documents)}
    return {"status": "unknown_entity", "entity": named, "document_ids": []}
