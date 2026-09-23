"""Offline regression tier: replay saved real-PaddleOCR output.

These tests assert the properties that make a fixture trustworthy — it is
stamped as a replay, it carries provenance, and it refuses to serve content
generated from a different image. They deliberately do NOT assert exact
recognition strings beyond a couple of stable anchors: the point is that
downstream code sees realistic OCR, not that PaddleOCR's output is frozen
forever.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.vision.ocr_engine import (
    FixtureOCREngine, MockOCREngine, OCR_FIXTURE_DIR, get_ocr_engine,
)

pytestmark = pytest.mark.fixture_ocr

ROOT = Path(__file__).resolve().parents[1]
DRAWINGS = ROOT / "data" / "drawings"


def test_get_ocr_engine_returns_fixture_engine():
    assert isinstance(get_ocr_engine("fixture"), FixtureOCREngine)


def test_fixture_result_is_labelled_as_replay_not_live_inference():
    """The stored payload says engine="paddleocr" (that is what produced it),
    but anything reading it back must see "paddleocr-fixture" — otherwise a
    report could present replayed output as this round's live run."""
    stored = json.loads((OCR_FIXTURE_DIR / "FAN-A13-02.json").read_text(encoding="utf-8"))
    assert stored["result"]["engine"] == "paddleocr"

    result = FixtureOCREngine().recognize(DRAWINGS / "FAN-A13-02.png")
    assert result.engine == "paddleocr-fixture"
    assert result.engine_available is True


def test_every_fixture_carries_full_provenance():
    required = {
        "image", "image_sha256", "generated_at", "paddleocr_version",
        "paddlepaddle_version", "models", "device", "enable_mkldnn",
    }
    fixtures = sorted(OCR_FIXTURE_DIR.glob("*.json"))
    assert fixtures, "no fixtures generated yet — run src.run_ocr_fixture_dump"
    for path in fixtures:
        provenance = json.loads(path.read_text(encoding="utf-8"))["provenance"]
        assert required <= set(provenance), f"{path.name} missing {required - set(provenance)}"


def test_fixture_without_provenance_is_rejected(tmp_path):
    """A provenance-less fixture is indistinguishable from a hand-written
    stub. Failing loudly is the whole reason this class exists."""
    (tmp_path / "FAN-A13-02.json").write_text(
        json.dumps({"result": {"text": "whatever", "blocks": []}}), encoding="utf-8")
    engine = FixtureOCREngine(fixture_dir=tmp_path)
    with pytest.raises(ValueError, match="no provenance"):
        engine.recognize(DRAWINGS / "FAN-A13-02.png")


def test_fixture_generated_from_a_different_image_is_rejected(tmp_path):
    """Stale fixtures are how a regression test quietly stops testing the
    thing it is named after."""
    payload = json.loads((OCR_FIXTURE_DIR / "FAN-A13-02.json").read_text(encoding="utf-8"))
    payload["provenance"]["image_sha256"] = "0" * 64
    (tmp_path / "FAN-A13-02.json").write_text(json.dumps(payload), encoding="utf-8")
    engine = FixtureOCREngine(fixture_dir=tmp_path)
    with pytest.raises(ValueError, match="different image"):
        engine.recognize(DRAWINGS / "FAN-A13-02.png")


def test_missing_fixture_raises_instead_of_scoring_zero():
    """Regression for a bug this change actually caused: preprocessed images
    (<name>.processed.png) had no fixtures, so a preprocessing A/B silently
    read them as empty and showed a 16-point drop that was entirely missing
    data, not a real effect. An absent fixture must be impossible to mistake
    for "OCR read this and found nothing"."""
    with pytest.raises(FileNotFoundError, match="run_ocr_fixture_dump"):
        FixtureOCREngine().recognize(DRAWINGS / "PUMP-02-01.png")


def test_preprocessed_variants_have_fixtures_so_the_ab_comparison_is_real():
    """Every image the preprocessing step can produce for the eval set needs
    its own fixture, or experiment B measures missing files rather than
    preprocessing."""
    for name in ["FAN-A22-01-heavyblur.processed", "FAN-A27-01-skewed.processed",
                 "FAN-A24-01.processed"]:
        assert (OCR_FIXTURE_DIR / f"{name}.json").exists(), f"missing fixture for {name}"


def test_fixture_bboxes_are_axis_aligned_four_number_boxes():
    """PaddleOCR returns 4-point polygons; this repo's downstream code (cell
    assignment in table_structure) reads bbox[0:4] as x0,y0,x1,y1. The
    conversion has to hold or cells get assigned to the wrong row."""
    result = FixtureOCREngine().recognize(DRAWINGS / "FAN-A13-02.png")
    assert result.blocks
    for block in result.blocks:
        assert len(block.bbox) == 4
        x0, y0, x1, y1 = block.bbox
        assert x0 < x1 and y0 < y1


def test_real_ocr_reads_a_field_the_handwritten_stub_deliberately_misreads():
    """Anchors the difference between the two tiers: the stub bakes in a
    realistic misread (QF-13 -> QF-I3) so validator conflict paths get
    exercised; the real engine reads it correctly. If these ever agree,
    one of the two has silently stopped being what it claims to be."""
    fixture = FixtureOCREngine().recognize(DRAWINGS / "FAN-A13-02.png")
    stub = MockOCREngine().recognize(DRAWINGS / "FAN-A13-02.png")
    assert "QF-13" in fixture.text
    assert "QF-I3" in stub.text and "QF-13" not in stub.text


def test_blurred_page_yields_no_text_and_zero_confidence():
    """The blur case is the honest-failure counterpart to the watermark case:
    real OCR finds nothing and says so with confidence 0.0, which is what
    lets confidence-based routing catch it."""
    result = FixtureOCREngine().recognize(DRAWINGS / "FAN-A22-01-heavyblur.png")
    assert result.blocks == []
    assert result.average_confidence == 0.0
