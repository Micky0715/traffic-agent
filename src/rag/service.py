"""The structured-BM25 vertical slice, as one callable service.

query -> parse -> BM25 -> eligibility -> evidence policy -> decision + refs

The CLI and the tests both go through this; neither re-implements the wiring.
A demo path that duplicates the logic proves only that the demo works.

Behind a feature flag: configs/rag.yaml `rag.mode` defaults to `legacy`, and
nothing in the existing pipeline calls this module. Turning it on is a config
change, not a code change.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rag.config import RagConfig, load_rag_config
from src.rag.eligibility import partition_candidates
from src.rag.ingest import (
    build_chunks_from_real_parsed_documents, entity_document_index,
)
from src.rag.policy import (
    Decision, EvidenceBundle, FieldEvidence, RequiredEvidencePolicy,
)
from src.rag.query import QueryPlan, parse_query, resolve_entity
from src.rag.retrieval import BM25Retriever
from src.vision.config import VisualFallbackConfig, load_config as load_vision_config
from src.vision.schemas import DocumentChunk


@dataclass
class QueryResult:
    query: str
    query_plan: Dict[str, Any]
    entity_resolution: Dict[str, Any]
    retrieved_candidates: List[Dict[str, Any]] = field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    eligible_evidence: List[Dict[str, Any]] = field(default_factory=list)
    decision: Dict[str, Any] = field(default_factory=dict)
    retrieved_candidate_count: int = 0
    eligible_evidence_count: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_plan": self.query_plan,
            "entity_resolution": self.entity_resolution,
            "retrieved_candidate_count": self.retrieved_candidate_count,
            "eligible_evidence_count": self.eligible_evidence_count,
            "retrieved_candidates": self.retrieved_candidates,
            "rejected_candidates": self.rejected_candidates,
            "eligible_evidence": self.eligible_evidence,
            "decision": self.decision,
            "latency_ms": round(self.latency_ms, 3),
        }


class StructuredRagService:
    """Corpus is built once and reused; queries are stateless."""

    def __init__(self, cfg: Optional[RagConfig] = None,
                 vision_cfg: Optional[VisualFallbackConfig] = None):
        self.cfg = cfg or load_rag_config()
        self.vision_cfg = vision_cfg or load_vision_config()
        self.chunks: List[DocumentChunk] = []
        self.stats = None
        self.retriever = BM25Retriever(self.cfg.retrieval)
        self.policy = RequiredEvidencePolicy(self.cfg.evidence_policy)
        self._entity_index: dict = {}

    def load_corpus(self) -> None:
        self.chunks, self.stats = build_chunks_from_real_parsed_documents(
            self.vision_cfg, self.cfg.chunking)
        self.retriever.index(self.chunks)
        self._entity_index = entity_document_index(self.chunks)

    # ------------------------------------------------------------------
    def answer(self, query: str, task_type: str = "drawing_field_query") -> QueryResult:
        started = time.perf_counter()
        plan = parse_query(query, self.cfg.query_parsing, task_type=task_type)
        resolution = resolve_entity(plan, self._entity_index)

        hits = self.retriever.search(plan.normalized_query,
                                     self.cfg.retrieval.sparse_top_k)
        resolved = resolution.get("entity") if resolution["status"] == "present" else None
        split = partition_candidates(
            plan, hits, resolved_entity=resolved,
            resolved_documents=resolution.get("document_ids"))

        result = QueryResult(
            query=query, query_plan=plan.to_dict(), entity_resolution=resolution,
            retrieved_candidates=[self._describe(h) for h in hits],
            rejected_candidates=[{**self._describe(r["candidate"]),
                                  "rejected_because": r["eligibility"].reasons}
                                 for r in split["rejected"]],
            eligible_evidence=[{**self._describe(r["candidate"]),
                                "matched_fields": r["eligibility"].matched_fields}
                               for r in split["eligible"]],
            retrieved_candidate_count=len(hits),
            eligible_evidence_count=len(split["eligible"]),
        )
        result.decision = self._decide(plan, resolution, split).to_dict()
        result.decision["retrieved_candidate_count"] = result.retrieved_candidate_count
        result.decision["eligible_evidence_count"] = result.eligible_evidence_count
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    # ------------------------------------------------------------------
    def _decide(self, plan: QueryPlan, resolution: Dict[str, Any],
                split: Dict[str, list]) -> Decision:
        # Questions only the user can settle are asked, never retried: no
        # amount of searching produces a device id the user did not give.
        if plan.entity_ambiguous:
            return Decision(
                decision="clarify", reason_codes=["ENTITY_AMBIGUOUS"],
                missing_fields=list(plan.required_fields),
                remediation=["ask_user"], routing_decision="clarify")
        if not plan.has_entity:
            return Decision(
                decision="clarify", reason_codes=["ENTITY_MISSING_FROM_QUERY"],
                missing_fields=list(plan.required_fields),
                remediation=["ask_user"], routing_decision="clarify")
        if not plan.required_fields:
            return Decision(
                decision="clarify", reason_codes=["TARGET_FIELD_NOT_SPECIFIED"],
                remediation=["ask_user"], routing_decision="clarify")

        # A device this corpus does not contain is not a retrieval failure to
        # keep trying at — it is an answer: we do not have that drawing.
        if resolution["status"] == "unknown_entity":
            return Decision(
                decision="abstain", reason_codes=["ENTITY_NOT_IN_CORPUS"],
                missing_fields=list(plan.required_fields),
                routing_decision="fallback")

        evidences: List[FieldEvidence] = []
        for record in split["eligible"]:
            chunk = record["chunk"]
            meta = chunk.metadata or {}
            evidences.append(FieldEvidence(
                field_name=chunk.field_name or "",
                value=chunk.field_value,
                source=chunk.parser_source or "unknown",
                document_id=chunk.document_id,
                page=chunk.page_start,
                bbox=chunk.source_bboxes[0] if chunk.source_bboxes else None,
                entity_id=chunk.entity_id,
                chunk_id=chunk.chunk_id,
                confidence=chunk.confidence,
                validation_status=meta.get("validation_status", "unknown"),
                conflict_with=meta.get("conflict"),
            ))

        # Retrieved-but-ineligible is not evidence. Handing the policy an empty
        # bundle here is the point: three candidates about other devices must
        # not become three reasons to answer.
        bundle = EvidenceBundle(
            task_type=plan.task_type,
            entity_id=resolution.get("entity"),
            requested_fields=list(plan.required_fields),
            fields=evidences,
            retrieved_chunks=[r["chunk"] for r in split["eligible"]],
        )
        decision = self.policy.evaluate(bundle)
        if not evidences and decision.decision not in {"clarify", "human_review"}:
            decision.decision = "abstain"
            decision.routing_decision = "fallback"
            if "NO_ELIGIBLE_EVIDENCE" not in decision.reason_codes:
                decision.reason_codes.append("NO_ELIGIBLE_EVIDENCE")
        return decision

    @staticmethod
    def _describe(hit) -> Dict[str, Any]:
        chunk = getattr(hit, "chunk", hit)
        return {
            "chunk_id": chunk.chunk_id,
            "content_type": chunk.content_type,
            "text": chunk.text,
            "document_id": chunk.document_id,
            "page": chunk.page_start,
            "bbox": chunk.source_bboxes[0] if chunk.source_bboxes else None,
            "entity_id": chunk.entity_id,
            "field_name": chunk.field_name,
            "field_value": chunk.field_value,
            "parser_source": chunk.parser_source,
            # Raw BM25 score and rank. A score, not a probability: BM25 is
            # unbounded and its scale depends on the corpus.
            "bm25_score": round(getattr(hit, "score", 0.0), 6),
            "bm25_rank": getattr(hit, "rank", None),
        }

    def capability_report(self) -> Dict[str, Any]:
        return {
            "retriever": {"name": self.retriever.name, "algorithm": "okapi_bm25",
                          "backend": "in_process",
                          "tokenizer": self.cfg.retrieval.tokenizer_name,
                          "stopwords_enabled": bool(self.cfg.retrieval.stopwords)},
            "elasticsearch": "not_connected",
            "milvus": "not_connected",
            "bge_embedding": "not_loaded",
            "reranker": "not_invoked",
            "vlm": "not_invoked",
            "dense_retriever": "not_used_in_this_mode",
            "human_reviewed_gold": False,
            "recall_at_5": "not_computed_no_independent_gold",
            "cache_used": ("ocr fixtures are a replay of a recorded real "
                           "PaddleOCR run; no live OCR in this path"),
        }
