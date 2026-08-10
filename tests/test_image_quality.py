from pathlib import Path

from src.vision.config import VisualFallbackConfig
from src.vision.image_quality import ImageQualityAnalyzer, decide_preprocess

DRAWINGS = Path(__file__).resolve().parents[1] / "data" / "drawings"
CFG = VisualFallbackConfig()
ANALYZER = ImageQualityAnalyzer()


def test_clean_drawing_has_reasonable_sharpness_and_contrast():
    metrics = ANALYZER.analyze(DRAWINGS / "FAN-A13-02.png")
    assert metrics.sharpness > 100  # crisp vector-drawn lines/text
    assert metrics.contrast > 20  # black lines on white background


def test_heavily_blurred_drawing_measures_lower_sharpness_than_clean():
    """Real, comparative assertion — FAN-A22-01-heavyblur.png was generated
    this session by downsampling+upsampling FAN-A22-01 specifically to be
    blurrier, so a real sharpness measurement must rank it below a clean
    drawing of the same kind. This isn't a hand-picked expected number."""
    clean = ANALYZER.analyze(DRAWINGS / "FAN-A13-02.png")
    blurry = ANALYZER.analyze(DRAWINGS / "FAN-A22-01-heavyblur.png")
    assert blurry.sharpness < clean.sharpness


def test_decide_preprocess_flags_denoise_for_blurry_image():
    blurry = ANALYZER.analyze(DRAWINGS / "FAN-A22-01-heavyblur.png")
    decision = decide_preprocess(blurry, CFG)
    assert decision.need_preprocess is True
    assert "denoise" in decision.operations


def test_decide_preprocess_is_empty_for_clean_image():
    clean = ANALYZER.analyze(DRAWINGS / "FAN-A13-02.png")
    decision = decide_preprocess(clean, CFG)
    assert decision.need_preprocess is False
    assert decision.operations == []


def test_analyze_raises_on_missing_file():
    import pytest
    with pytest.raises(FileNotFoundError):
        ANALYZER.analyze(DRAWINGS / "does-not-exist.png")
