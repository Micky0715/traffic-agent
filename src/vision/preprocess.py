from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.vision.schemas import ProcessedImage

# Fixed, sensible application order regardless of the order operations were
# requested in — e.g. deskewing after binarizing works better than before.
_ORDER = ["shadow_remove", "denoise", "binarize", "deskew", "perspective_correct", "resize"]


def denoise(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
    return cv2.fastNlMeansDenoising(image)


def binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) if image.ndim == 3 else binary


def deskew(image: np.ndarray, angle_deg: float = 0.0) -> np.ndarray:
    if abs(angle_deg) < 0.01:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def shadow_remove(image: np.ndarray) -> np.ndarray:
    """Background-division: estimate illumination via a large-kernel blur,
    divide it out, renormalize. Real cv2 math, not a stub."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR) if image.ndim == 3 else normalized


def perspective_correct(image: np.ndarray) -> np.ndarray:
    """Best-effort: find the largest 4-point contour and warp it to fill the
    frame. Real cv2 code, but untested against any real skewed-photo sample
    in this repo (only synthetic, already-rectangular drawings exist) — if
    no clean quadrilateral is found, returns the image unchanged rather than
    guessing at corners.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4 or cv2.contourArea(approx) < 0.3 * gray.shape[0] * gray.shape[1]:
        return image
    pts = approx.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")
    w = int(max(np.linalg.norm(ordered[0] - ordered[1]), np.linalg.norm(ordered[2] - ordered[3])))
    h = int(max(np.linalg.norm(ordered[0] - ordered[3]), np.linalg.norm(ordered[1] - ordered[2])))
    if w < 10 or h < 10:
        return image
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, matrix, (w, h))


def resize(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


_OPS = {
    "denoise": lambda img, angle: denoise(img),
    "binarize": lambda img, angle: binarize(img),
    "deskew": lambda img, angle: deskew(img, angle),
    "shadow_remove": lambda img, angle: shadow_remove(img),
    "perspective_correct": lambda img, angle: perspective_correct(img),
    "resize": lambda img, angle: resize(img),
}


def apply_preprocess(image_path: Path, operations: List[str], skew_angle_deg: float = 0.0) -> ProcessedImage:
    """Apply the requested operations in a fixed order and save to a new
    file under <original_dir>/processed/. The original is opened read-only;
    nothing in this function ever writes back to image_path.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")

    applied: List[str] = []
    for op in _ORDER:
        if op in operations:
            image = _OPS[op](image, skew_angle_deg)
            applied.append(op)

    processed_dir = image_path.parent / "processed"
    processed_dir.mkdir(exist_ok=True)
    processed_path = processed_dir / f"{image_path.stem}.processed{image_path.suffix}"
    cv2.imwrite(str(processed_path), image)

    return ProcessedImage(
        original_path=str(image_path),
        processed_path=str(processed_path),
        operations_applied=applied,
    )
