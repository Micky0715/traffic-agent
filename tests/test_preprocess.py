import shutil
from pathlib import Path

import cv2
import numpy as np

from src.vision.preprocess import apply_preprocess, binarize, denoise, deskew, perspective_correct

DRAWINGS = Path(__file__).resolve().parents[1] / "data" / "drawings"
SOURCE = DRAWINGS / "FAN-A13-02.png"


def test_apply_preprocess_never_writes_to_original_path(tmp_path):
    work_copy = tmp_path / "FAN-A13-02.png"
    shutil.copy(SOURCE, work_copy)
    original_bytes = work_copy.read_bytes()

    result = apply_preprocess(work_copy, ["denoise", "binarize"])

    assert work_copy.read_bytes() == original_bytes  # original untouched
    assert Path(result.processed_path).exists()
    assert result.processed_path != result.original_path


def test_apply_preprocess_records_operations_in_fixed_order(tmp_path):
    work_copy = tmp_path / "FAN-A13-02.png"
    shutil.copy(SOURCE, work_copy)

    # requested out of order; apply_preprocess normalizes to its fixed order
    result = apply_preprocess(work_copy, ["deskew", "denoise", "shadow_remove"], skew_angle_deg=2.0)
    assert result.operations_applied == ["shadow_remove", "denoise", "deskew"]


def test_apply_preprocess_with_no_operations_still_produces_a_readable_copy(tmp_path):
    work_copy = tmp_path / "FAN-A13-02.png"
    shutil.copy(SOURCE, work_copy)
    result = apply_preprocess(work_copy, [])
    assert result.operations_applied == []
    assert cv2.imread(result.processed_path) is not None


def test_denoise_preserves_shape_and_dtype():
    """Our synthetic drawings are near-noiseless vector-style images, so
    fastNlMeansDenoising may leave them almost unchanged — that's expected,
    not a bug. What must always hold is that the operation is shape/dtype
    preserving and doesn't error."""
    image = cv2.imread(str(SOURCE))
    denoised = denoise(image)
    assert denoised.shape == image.shape
    assert denoised.dtype == image.dtype


def test_binarize_produces_only_two_intensity_levels():
    image = cv2.imread(str(SOURCE))
    binary = binarize(image)
    gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    unique_values = set(np.unique(gray).tolist())
    assert unique_values.issubset({0, 255})


def test_deskew_zero_angle_is_a_no_op():
    image = cv2.imread(str(SOURCE))
    result = deskew(image, 0.0)
    assert np.array_equal(result, image)


def test_deskew_nonzero_angle_changes_the_image():
    image = cv2.imread(str(SOURCE))
    result = deskew(image, 5.0)
    assert not np.array_equal(result, image)


def test_perspective_correct_does_not_crash_on_a_flat_synthetic_drawing():
    """No real skewed-photo sample exists in this repo; this only asserts
    the function runs and returns a valid image, not that it improves
    anything (documented limitation, see preprocess.py docstring)."""
    image = cv2.imread(str(SOURCE))
    result = perspective_correct(image)
    assert result is not None
    assert result.ndim == image.ndim
