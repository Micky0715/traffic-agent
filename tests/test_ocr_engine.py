from pathlib import Path

from src.vision.ocr_engine import FixtureOCREngine, MockOCREngine, PaddleOCREngine, get_ocr_engine

DRAWINGS = Path(__file__).resolve().parents[1] / "data" / "drawings"


def test_mock_engine_returns_stub_result_for_known_image():
    engine = MockOCREngine()
    result = engine.recognize(DRAWINGS / "FAN-A13-02.png")
    assert result.engine == "mock"
    assert result.engine_available is True
    assert result.error is None
    assert "FAN-A13-02" in result.text


def test_mock_engine_stub_contains_realistic_ocr_error_not_ground_truth():
    """The stub deliberately misreads QF-13 as QF-I3 (digit 1 -> capital I)
    — if this ever regressed to always returning perfect ground truth, the
    Validator conflict-detection tests it feeds would stop testing anything
    real."""
    engine = MockOCREngine()
    result = engine.recognize(DRAWINGS / "FAN-A13-02.png")
    assert "QF-I3" in result.text
    assert "QF-13" not in result.text


def test_mock_engine_reports_missing_stub_without_crashing():
    engine = MockOCREngine()
    result = engine.recognize(DRAWINGS / "PUMP-02-01.png")  # no stub file authored for this one
    assert result.engine_available is True  # the mock engine itself is fine
    assert result.error is not None  # but it has nothing for this image
    assert result.text == ""


def test_stamp_occlusion_stub_shows_garbled_and_missing_fields():
    engine = MockOCREngine()
    result = engine.recognize(DRAWINGS / "FAN-A19-01.png")
    garbled_blocks = [b for b in result.blocks if b.confidence < 0.2]
    assert garbled_blocks  # at least one field is realistically unreadable under the stamp


def test_multi_device_stub_contains_a_ledger_mismatched_id():
    engine = MockOCREngine()
    result = engine.recognize(DRAWINGS / "FAN-MULTI-01.png")
    assert "A1B" in result.text  # OCR misread of A18, not in the asset ledger


def test_mock_engine_resolves_processed_image_path_back_to_same_stub():
    """The mock has no real pixel sensitivity, so it cannot show a
    preprocessing effect — it deliberately resolves a '<name>.processed.png'
    path back to the '<name>' stub rather than erroring or returning a
    different (fabricated) result."""
    engine = MockOCREngine()
    original = engine.recognize(DRAWINGS / "FAN-A13-02.png")
    processed = engine.recognize(Path("data/drawings/processed/FAN-A13-02.processed.png"))
    assert processed.text == original.text
    assert processed.error is None


def test_get_ocr_engine_defaults_to_mock():
    engine = get_ocr_engine("mock")
    assert isinstance(engine, MockOCREngine)


def test_get_ocr_engine_unknown_name_falls_back_to_mock():
    engine = get_ocr_engine("something_else")
    assert isinstance(engine, MockOCREngine)


def test_get_ocr_engine_selects_each_of_the_three_tiers():
    """mock / fixture / paddleocr are three different claims about where the
    text came from, and the config must be able to say which one."""
    assert isinstance(get_ocr_engine("mock"), MockOCREngine)
    assert isinstance(get_ocr_engine("fixture"), FixtureOCREngine)
    assert isinstance(get_ocr_engine("paddleocr"), PaddleOCREngine)


def test_paddle_engine_defaults_to_onednn_disabled():
    """Default False is not a style choice: oneDNN is what crashes inference
    on this machine, so an engine constructed without arguments has to be the
    one that works."""
    assert PaddleOCREngine()._enable_mkldnn is False
    assert PaddleOCREngine(enable_mkldnn=True)._enable_mkldnn is True
