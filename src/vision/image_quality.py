from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.vision.config import VisualFallbackConfig
from src.vision.schemas import ImageQualityMetrics, PreprocessDecision


def _load_gray(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _estimate_skew_angle(gray: np.ndarray) -> float:
    """minAreaRect over the thresholded foreground. Returns 0.0 for a page
    with too little foreground to estimate reliably (e.g. near-blank), not a
    guess — an unreliable estimate is worse than admitting there isn't one.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:  # normalize cv2's angle convention to [-45, 45]
        angle = 90 + angle
    return float(angle)


def _brightness_uniformity(gray: np.ndarray) -> float:
    """1.0 = perfectly even illumination. Splits the image into a coarse
    grid and compares the spread of block means to the overall mean.
    """
    h, w = gray.shape
    grid = 4
    block_means = []
    for i in range(grid):
        for j in range(grid):
            block = gray[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
            if block.size:
                block_means.append(float(block.mean()))
    if not block_means:
        return 1.0
    overall = float(gray.mean()) or 1.0
    spread = (max(block_means) - min(block_means)) / overall
    return max(0.0, 1.0 - spread)


class ImageQualityAnalyzer:
    """Real cv2-based measurement — every number here comes from actual
    pixels, nothing is hand-authored. This is one of the genuinely-real
    capabilities in this round (cv2 works fine on this machine; see
    src/vision/ocr_engine.py for the one that doesn't)."""

    def analyze(self, image_path: Path) -> ImageQualityMetrics:
        gray = _load_gray(image_path)
        return ImageQualityMetrics(
            sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            contrast=float(gray.std()),
            skew_angle_deg=_estimate_skew_angle(gray),
            brightness_uniformity=_brightness_uniformity(gray),
            mean_brightness=float(gray.mean()),
        )


def decide_preprocess(metrics: ImageQualityMetrics, cfg: VisualFallbackConfig) -> PreprocessDecision:
    """Pick only the operations this specific image actually needs.

    Deliberately not a fixed denoise->binarize->deskew->perspective->superres
    stack: each operation is gated on the specific measurement it addresses,
    matching the "preprocess by need" principle in the plan. Superres/resize
    and perspective_correct are defined in preprocess.py but are not
    auto-triggered here — this repo has no real skewed-photo samples and no
    reliable signal for "would resolution enhancement actually help", so
    auto-deciding those would be guessing, not measuring.
    """
    pp = cfg.preprocess
    operations: List[str] = []
    reasons: List[str] = []

    if metrics.sharpness < pp.min_sharpness_for_skip_denoise:
        operations.append("denoise")
        reasons.append("low_sharpness")
    if metrics.contrast < pp.min_contrast_for_skip_binarize:
        operations.append("binarize")
        reasons.append("low_contrast")
    if abs(metrics.skew_angle_deg) > pp.skew_angle_threshold_deg:
        operations.append("deskew")
        reasons.append("skew_angle_high")
    if metrics.brightness_uniformity < pp.min_brightness_uniformity:
        operations.append("shadow_remove")
        reasons.append("uneven_illumination")

    return PreprocessDecision(need_preprocess=bool(operations), operations=operations, reasons=reasons)
