"""Hybrid retrieval: lexical + dense, fused by RRF, expanded by parent.

What is real here and what is not, stated once so no report has to guess:

  BM25 is the real Okapi BM25 scoring function over an in-process index. It is
  not Elasticsearch. Swapping in ES means writing an adapter against the
  Retriever protocol; nothing else changes.

  The dense side is a protocol with a deterministic offline implementation. No
  embedding model is installed in this environment, and a retriever that
  invented similarity scores would make every fusion number meaningless while
  looking exactly like a working one. The offline implementation scores by a
  declared, inspectable rule and says so through `is_real_model = False`.

  No reranker model is installed. The hook exists; running it is reported as
  not_invoked rather than skipped silently.

RRF scores are rank-fusion weights, not probabilities. With k=60 a document
ranked first by both retrievers scores 2/61 ≈ 0.0328 — a number that says
"agreed on by both at the top", not "3% confident". Any threshold expressed as
"RRF > 0.8" is meaningless.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

from src.rag.config import RetrievalConfig
from src.vision.schemas import DocumentChunk

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*|[一-鿿]")


def tokenize(text: str, stopwords: Optional[Sequence[str]] = None) -> List[str]:
    """Latin/code runs kept whole, CJK split per character.

    Codes must survive as single tokens — splitting "FAN-A13-02" into three
    pieces is precisely how an exact-identifier query stops being exact. CJK is
    per-character because no segmenter is installed here and none is installed
    automatically; character unigrams are a defensible fallback rather than a
    guessed word list.

    Stopwords are removed when supplied. Without them an unrelated Chinese
    question shares 的/不 with essentially every Chinese document and therefore
    scores above zero against all of them. That is a tokenizer property, not a
    per-query fix — the real defence against answering from irrelevant hits is
    the eligibility gate in src/rag/eligibility.py.
    """
    tokens = [t.lower() for t in _TOKEN.findall(text or "")]
    if stopwords:
        blocked = {w.lower() for w in stopwords}
        tokens = [t for t in tokens if t not in blocked]
    return tokens


@dataclass
class RetrievalHit:
    chunk: DocumentChunk
    score: float
    rank: int
    source: str          # "sparse" | "dense"


@dataclass
class FusedHit:
    """One candidate after fusion, carrying where it came from.

    Per-retriever ranks are kept so a result set is auditable: "this came top
    from BM25 and 14th from dense" is a reviewable statement; a single fused
    number is not.
    """

    chunk: DocumentChunk
    rrf_score: float
    sparse_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    dense_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rerank_status: str = "not_invoked"
    parent_context: Optional[str] = None
    fused_rank: int = 0


class Retriever(Protocol):
    name: str
    is_real_service: bool

    def index(self, chunks: Sequence[DocumentChunk]) -> None: ...
    def search(self, query: str, top_k: int) -> List[RetrievalHit]: ...


# ---------------------------------------------------------------------------
# lexical
# ---------------------------------------------------------------------------

class BM25Retriever:
    """Okapi BM25 over an in-process index. Real scoring, not a real service."""

    name = "bm25_local"
    is_real_service = False   # in-process; not Elasticsearch

    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self._chunks: List[DocumentChunk] = []
        self._docs: List[List[str]] = []
        self._freqs: List[Counter] = []
        self._df: Counter = Counter()
        self._avg_len = 0.0

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self._docs = [tokenize(c.text, self.cfg.stopwords) for c in self._chunks]
        self._freqs = [Counter(doc) for doc in self._docs]
        self._df = Counter()
        for freq in self._freqs:
            self._df.update(freq.keys())
        self._avg_len = (sum(len(d) for d in self._docs) / len(self._docs)
                         if self._docs else 0.0)

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        # Standard BM25 idf with the +0.5 smoothing; floored at 0 so a term in
        # almost every document cannot contribute negatively.
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def search(self, query: str, top_k: int) -> List[RetrievalHit]:
        terms = tokenize(query, self.cfg.stopwords)
        if not terms or not self._docs:
            return []
        k1, b = self.cfg.bm25_k1, self.cfg.bm25_b
        scored = []
        for index, freq in enumerate(self._freqs):
            length = len(self._docs[index]) or 1
            score = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denom = tf + k1 * (1 - b + b * length / (self._avg_len or 1))
                score += self._idf(term) * tf * (k1 + 1) / denom
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [RetrievalHit(chunk=self._chunks[i], score=s, rank=rank + 1,
                             source="sparse")
                for rank, (s, i) in enumerate(scored[:top_k])]


# ---------------------------------------------------------------------------
# dense
# ---------------------------------------------------------------------------

class OfflineDeterministicDenseRetriever:
    """Stand-in for a dense retriever, for use when no embedding model exists.

    Scores by token-set overlap weighted toward rarer tokens. That is a real,
    declared, reproducible rule — NOT a random number and NOT a pre-written
    answer. It behaves like a semantic retriever only in the weak sense that it
    is recall-oriented and unordered; it does not model synonymy, and the
    reports say so.

    `is_real_model = False` so no caller can mistake it for one.
    """

    name = "dense_offline_deterministic"
    is_real_service = False
    is_real_model = False

    def __init__(self, cfg: RetrievalConfig, synonyms: Optional[Dict[str, List[str]]] = None):
        self.cfg = cfg
        # Declared synonym expansion, from configuration — not learned, and
        # visible in the report so nobody reads it as semantic capability.
        self.synonyms = synonyms or {}
        self._chunks: List[DocumentChunk] = []
        self._sets: List[set] = []
        self._df: Counter = Counter()

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self._sets = [set(tokenize(c.text)) for c in self._chunks]
        self._df = Counter()
        for tokens in self._sets:
            self._df.update(tokens)

    def _expand(self, terms: Iterable[str]) -> set:
        expanded = set()
        for term in terms:
            expanded.add(term)
            expanded.update(t.lower() for t in self.synonyms.get(term, []))
        return expanded

    def search(self, query: str, top_k: int) -> List[RetrievalHit]:
        terms = self._expand(tokenize(query))
        if not terms or not self._sets:
            return []
        n = len(self._sets)
        scored = []
        for index, tokens in enumerate(self._sets):
            shared = terms & tokens
            if not shared:
                continue
            weight = sum(math.log(1 + n / (1 + self._df.get(t, 0))) for t in shared)
            score = weight / math.sqrt(len(tokens) or 1)
            scored.append((score, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [RetrievalHit(chunk=self._chunks[i], score=s, rank=rank + 1,
                             source="dense")
                for rank, (s, i) in enumerate(scored[:top_k])]


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    sparse: Sequence[RetrievalHit],
    dense: Sequence[RetrievalHit],
    cfg: RetrievalConfig,
) -> List[FusedHit]:
    """RRF(d) = sum_i 1 / (k + rank_i(d)).

    Deduplicated by chunk_id, NOT by text: two rows of one table can read
    similarly while describing different devices, and merging them would
    silently drop one device's data.
    """
    fused: Dict[str, FusedHit] = {}

    def key(hit: RetrievalHit) -> str:
        return hit.chunk.chunk_id or f"{hit.chunk.document_id}:{hash(hit.chunk.text)}"

    for hit in sparse:
        entry = fused.setdefault(key(hit), FusedHit(chunk=hit.chunk, rrf_score=0.0))
        entry.sparse_rank, entry.sparse_score = hit.rank, hit.score
        entry.rrf_score += 1.0 / (cfg.rrf_k + hit.rank)
    for hit in dense:
        entry = fused.setdefault(key(hit), FusedHit(chunk=hit.chunk, rrf_score=0.0))
        entry.dense_rank, entry.dense_score = hit.rank, hit.score
        entry.rrf_score += 1.0 / (cfg.rrf_k + hit.rank)

    ordered = sorted(fused.values(),
                     key=lambda h: (-h.rrf_score, h.chunk.chunk_id))
    for rank, hit in enumerate(ordered, start=1):
        hit.fused_rank = rank
    return ordered[:cfg.final_top_k]


def expand_parent_context(hits: Sequence[FusedHit],
                          chunks: Sequence[DocumentChunk]) -> List[FusedHit]:
    """Attach each hit's own parent chunk — table title, section heading.

    Strictly the declared parent, never "the N chunks around it": neighbouring
    chunks in a table are OTHER DEVICES, and splicing them in is how one
    device's figures end up attributed to another.
    """
    by_id = {c.chunk_id: c for c in chunks if c.chunk_id}
    parents = {c.parent_id: c for c in chunks
               if c.content_type in {"table_parent", "section"}}
    for hit in hits:
        parent = parents.get(hit.chunk.parent_id) or by_id.get(hit.chunk.parent_id)
        if parent is not None and parent.chunk_id != hit.chunk.chunk_id:
            hit.parent_context = parent.text
    return list(hits)


@dataclass
class HybridRetriever:
    """Sparse + dense + RRF + optional parent expansion."""

    cfg: RetrievalConfig
    sparse: BM25Retriever
    dense: object
    chunks: List[DocumentChunk] = field(default_factory=list)

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        self.sparse.index(self.chunks)
        self.dense.index(self.chunks)

    def search(self, query: str) -> List[FusedHit]:
        sparse_hits = self.sparse.search(query, self.cfg.sparse_top_k)
        dense_hits = self.dense.search(query, self.cfg.dense_top_k)
        fused = reciprocal_rank_fusion(sparse_hits, dense_hits, self.cfg)
        if self.cfg.parent_expansion:
            fused = expand_parent_context(fused, self.chunks)
        return fused

    def capability_report(self) -> Dict[str, object]:
        """What actually ran. Consumed by the audit report so no number is
        presented as coming from a service that was never contacted."""
        return {
            "sparse": {"name": self.sparse.name, "real_service": self.sparse.is_real_service,
                       "algorithm": "okapi_bm25", "backend": "in_process"},
            "dense": {"name": getattr(self.dense, "name", "unknown"),
                      "real_service": getattr(self.dense, "is_real_service", False),
                      "real_model": getattr(self.dense, "is_real_model", False)},
            "reranker": {"status": "not_invoked",
                         "reason": "no reranker model installed in this environment"},
            "elasticsearch": {"status": "not_connected"},
            "milvus": {"status": "not_connected"},
        }


def build_hybrid_retriever(cfg: RetrievalConfig,
                           synonyms: Optional[Dict[str, List[str]]] = None) -> HybridRetriever:
    return HybridRetriever(cfg=cfg, sparse=BM25Retriever(cfg),
                           dense=OfflineDeterministicDenseRetriever(cfg, synonyms))
