"""Build the retrieval corpus from REAL parsed drawing output.

Source is data/ocr_fixtures/*.json — saved output of a real PaddleOCR 3.7.0 run,
each file stamped with the image sha256, library and model versions, device and
generation time. That is real recognition output, replayed; it is not
hand-written mock text, and it is not live inference either. Both halves of
that sentence matter and both are carried into every report.

Hand-written mocks (data/ocr_stub, data/mock_drawings.json) stay available for
unit tests and are counted separately, because a structural unit test and an
effectiveness measurement need different kinds of input and mixing their counts
makes the second one unreadable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.rag.chunking import build_table_chunks, content_hash, stable_chunk_id
from src.rag.config import ChunkingConfig
from src.vision.config import VisualFallbackConfig
from src.vision.field_completeness import evaluate_field_completeness
from src.vision.ocr_engine import OCR_FIXTURE_DIR, FixtureOCREngine
from src.vision.schemas import DocumentChunk
from src.vision.table_structure import detect_table_structure
from src.vision.value_validation import (
    field_pairs_from_completeness, normalize_value, validate_field_values,
)

ROOT = Path(__file__).resolve().parents[2]
DRAWINGS_DIR = ROOT / "data" / "drawings"


@dataclass
class CorpusStats:
    """Counts kept apart by provenance.

    A single "chunk count" that blends replayed real recognition with
    hand-authored fixtures cannot be read: the reader cannot tell which number
    supports an effectiveness claim and which only supports a structural one.
    """

    real_documents: int = 0
    real_chunks: int = 0
    synthetic_chunks: int = 0
    mock_chunks: int = 0
    field_pairs: int = 0
    pairs_with_bbox: int = 0
    documents: List[Dict[str, object]] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return self.real_chunks + self.synthetic_chunks + self.mock_chunks


def _fixture_provenance(stem: str) -> Dict[str, object]:
    path = OCR_FIXTURE_DIR / f"{stem}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("provenance", {})


def build_chunks_from_real_parsed_documents(
    vision_cfg: VisualFallbackConfig,
    chunk_cfg: ChunkingConfig,
    *,
    fixture_dir: Optional[Path] = None,
    drawings_dir: Optional[Path] = None,
    parser_source: str = "ocr",
) -> tuple[List[DocumentChunk], CorpusStats]:
    """Real OCR fixtures -> field pairs -> DocumentChunks.

    The field pairs come from the frozen vision pipeline
    (field_completeness.evaluate_field_completeness), so label->value pairing
    and its bbox are exactly what the parser produced — this module does not
    re-derive them and cannot disagree with the pipeline about which block
    holds a field's value.

    parser_source defaults to "ocr" because these fixtures ARE OCR output. A
    VLM-produced corpus must pass "vlm": presenting a model's reading as text
    read off the page is the one provenance error that cannot be undone
    downstream.
    """
    fixture_dir = fixture_dir or OCR_FIXTURE_DIR
    drawings_dir = drawings_dir or DRAWINGS_DIR
    engine = FixtureOCREngine(fixture_dir=fixture_dir)
    stats = CorpusStats()
    chunks: List[DocumentChunk] = []

    def normalize(value: str) -> str:
        return normalize_value(value, vision_cfg.value_validation.hyphen_variants,
                               vision_cfg.value_validation.collapse_spaces_around_hyphen)

    for fixture_path in sorted(fixture_dir.glob("*.json")):
        stem = fixture_path.stem
        if stem.endswith(".processed"):
            # Preprocessed variants are the same page re-rendered; indexing both
            # would double every field and let one page outvote the others.
            continue
        image = drawings_dir / f"{stem}.png"
        if not image.exists():
            stats.skipped.append({"document": stem, "reason": "source image missing"})
            continue

        ocr = engine.recognize(image)
        table = detect_table_structure(image, ocr_result=ocr)
        completeness = evaluate_field_completeness(
            ocr, vision_cfg.drawing_types, vision_cfg.field_completeness,
            table=table, min_table_confidence=vision_cfg.min_table_confidence)
        validation = validate_field_values(
            field_pairs_from_completeness(completeness.field_pairs),
            vision_cfg.value_validation,
            vision_cfg.drawing_types.get(completeness.drawing_type))

        document_id = stem
        page = 1                     # one drawing per image file
        drawing_no = next((p.raw_value for p in completeness.field_pairs
                           if p.field_name == "图号"), None)
        parent_id = f"{document_id}:p{page}:drawing"
        invalid = set(validation.invalid_fields)

        document_chunks: List[DocumentChunk] = []
        for pair in completeness.field_pairs:
            stats.field_pairs += 1
            if pair.bbox:
                stats.pairs_with_bbox += 1
            if not pair.raw_value.strip():
                continue
            qualified = f"{pair.entity_id}{pair.field_name}" if pair.entity_id \
                else pair.field_name
            # Both forms kept: the raw one is the evidence of what the page
            # says, the normalized one is what matching needs.
            raw_value = pair.raw_value
            normalized = normalize(raw_value)
            prefix = f"图号 {drawing_no}；" if drawing_no else ""
            entity_bit = f"设备 {pair.entity_id}；" if pair.entity_id else ""
            text = f"{prefix}{entity_bit}{pair.field_name} {raw_value}"

            document_chunks.append(DocumentChunk(
                text=text,
                parent_id=parent_id,
                page_number=page,
                document_id=document_id,
                content_type="drawing_field",
                chunk_id=stable_chunk_id(document_id, "drawing_field",
                                         qualified, raw_value),
                source_hash=content_hash(text),
                page_start=page, page_end=page,
                entity_id=pair.entity_id,
                field_name=pair.field_name,
                field_value=raw_value,
                source_bboxes=[list(pair.bbox)] if pair.bbox else [],
                parser_source=parser_source,
                confidence=pair.confidence,
                structure_uncertain=completeness.group_cardinality_uncertain,
                metadata={
                    "drawing_no": drawing_no,
                    "drawing_type": completeness.drawing_type,
                    "normalized_value": normalized,
                    # invalid travels with the chunk so eligibility can refuse
                    # it without re-running validation.
                    "validation_status": "invalid" if qualified in invalid else "valid",
                    "ocr_engine": ocr.engine,
                    "provenance": _fixture_provenance(stem),
                },
            ))

        # A parent so a retrieved field can be expanded to its drawing without
        # splicing in a neighbouring device's chunks.
        if document_chunks:
            header = f"图纸 {drawing_no or document_id}；类型 {completeness.drawing_type}"
            document_chunks.insert(0, DocumentChunk(
                text=header,
                parent_id=parent_id,
                page_number=page,
                document_id=document_id,
                content_type="drawing_metadata",
                chunk_id=stable_chunk_id(document_id, "drawing_metadata",
                                         "header", header),
                source_hash=content_hash(header),
                page_start=page, page_end=page,
                parser_source=parser_source,
                confidence=ocr.average_confidence,
                metadata={"drawing_no": drawing_no,
                          "drawing_type": completeness.drawing_type},
            ))

        chunks.extend(document_chunks)
        stats.real_documents += 1
        stats.real_chunks += len(document_chunks)
        stats.documents.append({
            "document_id": document_id,
            "drawing_no": drawing_no,
            "drawing_type": completeness.drawing_type,
            "chunk_count": len(document_chunks),
            "ocr_engine": ocr.engine,
            "ocr_confidence": round(ocr.average_confidence, 4),
            "invalid_fields": sorted(invalid),
            "entities": sorted({c.entity_id for c in document_chunks if c.entity_id}),
        })

    return chunks, stats


def entity_document_index(chunks: Sequence[DocumentChunk]) -> Dict[str, set]:
    """Every string the corpus can be addressed by -> the documents it names.

    Mapping to DOCUMENTS, not just collecting names, is what lets a page-level
    field be attributed correctly. On a single-device wiring diagram the
    fields carry no entity_id — they belong to the page — so "A13风机的电机编号"
    can only be resolved by first resolving A13风机 to its drawing and then
    accepting that drawing's page-level fields. Matching entity strings alone
    rejects them, which is exactly what it did before this index existed.
    """
    index: Dict[str, set] = {}

    def add(name: Optional[str], document_id: str) -> None:
        if name:
            index.setdefault(name, set()).add(document_id)

    for chunk in chunks:
        add(chunk.entity_id, chunk.document_id)
        add((chunk.metadata or {}).get("drawing_no"), chunk.document_id)
        if chunk.field_name in {"电机编号", "断路器编号", "控制柜编号", "设备编号"}:
            add(chunk.field_value, chunk.document_id)
        # A drawing's title names the equipment it documents ("A13风机接线图"),
        # and users address drawings that way.
        if chunk.field_name == "名称":
            add(chunk.field_value, chunk.document_id)
        add(chunk.document_id, chunk.document_id)
    return index


def known_entities(chunks: Sequence[DocumentChunk]) -> set:
    return set(entity_document_index(chunks))
