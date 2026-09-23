"""Live-inference tier: actually runs PaddleOCR. Deselected by default.

    pytest -m real_ocr

Needs `pip install paddleocr==3.7.0 paddlepaddle==3.3.1` and several hundred
MB of model weights, and takes ~35s per image. Everything here is about the
engine adapter being correct against a real model — the properties that a
stub or a fixture cannot prove, because both of them are frozen output.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.vision.ocr_engine import PaddleOCREngine

pytestmark = pytest.mark.real_ocr

ROOT = Path(__file__).resolve().parents[1]
DRAWINGS = ROOT / "data" / "drawings"


@pytest.fixture(scope="module")
def engine() -> PaddleOCREngine:
    # One engine per module: initialization alone costs ~11s.
    return PaddleOCREngine(enable_mkldnn=False)


def test_live_inference_succeeds_with_onednn_disabled(engine):
    """The regression this whole change exists for: with oneDNN on, predict()
    raises ConvertPirAttribute2RuntimeAttribute from onednn_instruction.cc
    and the engine reports engine_available=False. With it off, inference
    completes."""
    result = engine.recognize(DRAWINGS / "FAN-A13-02.png")
    assert result.engine == "paddleocr"
    assert result.engine_available is True, f"live inference failed: {result.error}"
    assert result.error is None
    assert result.blocks
    assert result.average_confidence > 0.9


def test_live_inference_reads_the_expected_title_block_fields(engine):
    result = engine.recognize(DRAWINGS / "FAN-A13-02.png")
    for expected in ["FAN-A13-02", "M-13", "QF-13", "FAN-CAB-3"]:
        assert expected in result.text, f"{expected} not found in: {result.text!r}"


def test_live_bboxes_are_converted_to_axis_aligned_boxes(engine):
    """PaddleOCR hands back 4-point polygons (8 numbers). Flattening them
    would give downstream code a zero-height box; poly_to_xyxy must collapse
    them to real [x0, y0, x1, y1] bounds."""
    result = engine.recognize(DRAWINGS / "FAN-A13-02.png")
    for block in result.blocks:
        assert len(block.bbox) == 4
        x0, y0, x1, y1 = block.bbox
        assert x1 > x0 and y1 > y0


def test_unreadable_page_fails_loudly_rather_than_inventing_text(engine):
    """A heavily blurred page must come back empty with confidence 0.0, not
    with a hallucinated low-confidence guess — confidence-based routing
    depends on that being honest."""
    result = engine.recognize(DRAWINGS / "FAN-A22-01-heavyblur.png")
    assert result.engine_available is True
    assert result.blocks == []
    assert result.average_confidence == 0.0


def test_missing_image_is_reported_not_raised(engine):
    result = engine.recognize(DRAWINGS / "does-not-exist.png")
    assert result.engine_available is False
    assert result.error
