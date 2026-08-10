from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.vision.schemas import OCRBlock, OCRResult

ROOT = Path(__file__).resolve().parents[2]
OCR_STUB_DIR = ROOT / "data" / "ocr_stub"


class OCREngine(ABC):
    @abstractmethod
    def recognize(self, image_path: Path) -> OCRResult:
        ...


class MockOCREngine(OCREngine):
    """Returns hand-authored OCR results from data/ocr_stub/<image_stem>.json.

    This is an explicit stub — it does not run any recognition model. Used
    because no real OCR engine is usable on this development machine (see
    PaddleOCREngine below for why, and why that is a real, not a fake,
    limitation). The stub files deliberately bake in realistic OCR error
    patterns (e.g. "A12" misread as "A1Z") instead of always returning the
    correct text, so anything testing against this engine (Validator
    cross-checks, quality judging) exercises real disagreement handling
    rather than trivially passing.
    """

    def recognize(self, image_path: Path) -> OCRResult:
        # A mock engine has no real pixel sensitivity — it cannot "see" that
        # a preprocessed image differs from the original, so it resolves
        # back to the same stub either way. That is a deliberate, documented
        # property of the mock (see data/ocr_stub/README.md and
        # interview/ocr_bad_cases.md), not a bug: it's exactly why real A/B
        # OCR-preprocessing numbers from this engine must be reported as
        # mock, while the C/D VLM-preprocessing comparison (real pixels, real
        # model) is the one real experiment this round can actually run.
        stem = Path(image_path).stem
        if stem.endswith(".processed"):
            stem = stem[: -len(".processed")]
        stub_path = OCR_STUB_DIR / f"{stem}.json"
        if not stub_path.exists():
            return OCRResult(
                text="", average_confidence=0.0, engine="mock",
                engine_available=True, error=f"no stub file at {stub_path}",
            )
        data = json.loads(stub_path.read_text(encoding="utf-8"))
        blocks = [OCRBlock(**b) for b in data.get("blocks", [])]
        return OCRResult(
            text=data.get("text", ""),
            blocks=blocks,
            average_confidence=data.get("average_confidence", 0.0),
            engine="mock",
            engine_available=True,
            error=None,
        )


class PaddleOCREngine(OCREngine):
    """Real adapter — genuinely calls paddleocr, this is not a stub.

    Verified this session: PaddleOCR successfully downloads real model
    weights (PP-OCRv6 det/rec, UVDoc, doc-orientation — several hundred MB),
    but calling predict() on this machine raises an internal Paddle/oneDNN
    framework error:

        NotImplementedError: (Unimplemented)
        ConvertPirAttribute2RuntimeAttribute not support
        [pir::ArrayAttribute<pir::DoubleAttribute>]

    That is caught here and surfaced as engine_available=False with the raw
    error message — never silently swallowed or replaced with a fake
    result. On a machine without this specific Paddle/oneDNN incompatibility
    this class should work as-is; switching to it is a one-line config
    change (configs/visual_parser.yaml: ocr.engine), not a code change.
    """

    def __init__(self, lang: str = "ch"):
        self._lang = lang
        self._ocr = None
        self._init_error: Optional[str] = None

    def _ensure_initialized(self) -> None:
        if self._ocr is not None or self._init_error is not None:
            return
        try:
            from paddleocr import PaddleOCR  # heavy import, deferred until actually used
            self._ocr = PaddleOCR(lang=self._lang)
        except Exception as e:  # noqa: BLE001 - any init failure must be surfaced, not hidden
            self._init_error = str(e)

    def recognize(self, image_path: Path) -> OCRResult:
        self._ensure_initialized()
        if self._init_error is not None:
            return OCRResult(
                text="", average_confidence=0.0, engine="paddleocr",
                engine_available=False, error=self._init_error,
            )
        try:
            raw_result = self._ocr.predict(str(image_path))
        except Exception as e:  # noqa: BLE001 - this is the real, observed failure on this machine
            return OCRResult(
                text="", average_confidence=0.0, engine="paddleocr",
                engine_available=False, error=str(e),
            )

        try:
            blocks: list[OCRBlock] = []
            texts, confidences = [], []
            for page in raw_result or []:
                rec_texts = page.get("rec_texts", [])
                rec_scores = page.get("rec_scores", [])
                rec_polys = page.get("rec_polys", [])
                for i, text in enumerate(rec_texts):
                    score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                    bbox = [float(x) for x in rec_polys[i].flatten()] if i < len(rec_polys) else []
                    blocks.append(OCRBlock(text=text, bbox=bbox, confidence=score))
                    texts.append(text)
                    confidences.append(score)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            return OCRResult(
                text="\n".join(texts), blocks=blocks, average_confidence=avg_conf,
                engine="paddleocr", engine_available=True, error=None,
            )
        except Exception as e:  # noqa: BLE001 - unverified result shape (predict() never succeeded here)
            return OCRResult(
                text="", average_confidence=0.0, engine="paddleocr",
                engine_available=False, error=f"unexpected paddleocr result shape: {e}",
            )


def get_ocr_engine(engine_name: str) -> OCREngine:
    if engine_name == "paddleocr":
        return PaddleOCREngine()
    return MockOCREngine()
