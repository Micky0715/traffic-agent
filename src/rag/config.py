"""Configuration for structured chunking, retrieval and evidence policy.

Every threshold lives here or in configs/rag.yaml. Nothing in src/rag may
hard-code a number that decides behaviour — a magic constant buried in code is
a threshold nobody can find, review or change per deployment.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "rag.yaml"


class ChunkingConfig(BaseModel):
    section_max_tokens: int = 600
    section_overlap_tokens: int = 80
    # Cross-page table continuation. Hard signals must ALL hold; supporting
    # signals are counted. Column count alone is deliberately not enough.
    continuation_required_signals: List[str] = Field(
        default_factory=lambda: ["adjacent_pages", "header_compatible",
                                 "column_count_compatible"])
    continuation_supporting_signals: List[str] = Field(
        default_factory=lambda: ["same_section", "continuation_marker",
                                 "previous_unterminated"])
    continuation_min_supporting: int = 1


class QueryParsingConfig(BaseModel):
    """Canonical field names and their surface forms.

    Configured, never learned and never completed by a model: a field the user
    did not ask for, filled in by a guess, produces a confident answer to a
    question nobody put.
    """

    canonical_fields: List[str] = Field(default_factory=list)
    field_synonyms: Dict[str, List[str]] = Field(default_factory=dict)

    def all_field_surfaces(self) -> List[str]:
        surfaces = list(self.canonical_fields)
        for canonical, aliases in self.field_synonyms.items():
            surfaces.extend(aliases)
        return sorted(set(surfaces))


class RetrievalConfig(BaseModel):
    # Which tokenizer produced the index, recorded so a report can say how
    # text was split rather than leaving the reader to assume.
    tokenizer_name: str = "latin_code_runs_plus_cjk_unigram"
    # CJK function words. Without them an unrelated question shares 的/不 with
    # almost any Chinese text and scores above zero against all of it.
    stopwords: List[str] = Field(default_factory=list)
    # Deliberately NOT applied. Calibrating a score floor needs a dev set;
    # picking one because a single query misbehaved is guesswork. Recorded as
    # a config item so calibration has somewhere to land.
    uncalibrated_min_bm25_score: Optional[float] = None
    uncalibrated_status: str = "uncalibrated"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    sparse_top_k: int = 20
    dense_top_k: int = 20
    # RRF constant. Larger k flattens the contribution of top ranks.
    rrf_k: int = 60
    final_top_k: int = 10
    rerank_top_k: int = 10
    parent_expansion: bool = True


class EvidencePolicyConfig(BaseModel):
    # Per task type: which evidence must be present before an answer may be
    # produced without a human.
    required_evidence: Dict[str, List[str]] = Field(default_factory=dict)
    high_risk_fields: List[str] = Field(default_factory=list)
    max_remediation_rounds: int = 2
    min_field_confidence: float = 0.5


class RagConfig(BaseModel):
    # Feature flag. `legacy` keeps every existing entry point behaving exactly
    # as before; `structured_bm25` is opt-in and reached only through
    # src/run_structured_rag_query.py.
    mode: Literal["legacy", "structured_bm25"] = "legacy"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    query_parsing: QueryParsingConfig = Field(default_factory=QueryParsingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evidence_policy: EvidencePolicyConfig = Field(default_factory=EvidencePolicyConfig)


@lru_cache(maxsize=4)
def load_rag_config(path: str | None = None) -> RagConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"rag config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return RagConfig(
        mode=(raw.get("rag") or {}).get("mode", "legacy"),
        chunking=ChunkingConfig(**(raw.get("chunking") or {})),
        query_parsing=QueryParsingConfig(**(raw.get("query_parsing") or {})),
        retrieval=RetrievalConfig(**(raw.get("retrieval") or {})),
        evidence_policy=EvidencePolicyConfig(**(raw.get("evidence_policy") or {})),
    )
