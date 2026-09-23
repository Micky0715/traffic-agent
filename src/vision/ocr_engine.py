from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

from src.vision.schemas import OCRBlock, OCRResult

ROOT = Path(__file__).resolve().parents[2]
OCR_STUB_DIR = ROOT / "data" / "ocr_stub"
OCR_FIXTURE_DIR = ROOT / "data" / "ocr_fixtures"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def poly_to_xyxy(poly: Any) -> List[float]:
    """Collapse PaddleOCR's 4-point quadrilateral into the axis-aligned
    [x0, y0, x1, y1] box the rest of this repo assumes.

    This matters: an earlier version flattened the polygon to 8 numbers, and
    everything downstream that reads bbox[0:4] (table_structure's cell
    assignment, for one) would have silently interpreted the first two
    CORNERS as a box — giving a y-range of zero height. It was never caught
    because predict() had never successfully returned on this machine, so no
    real bbox ever reached that code.
    """
    points = [(float(x), float(y)) for x, y in (poly.tolist() if hasattr(poly, "tolist") else poly)]
    if not points:
        return []
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return [min(xs), min(ys), max(xs), max(ys)]


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


class FixtureOCREngine(OCREngine):
    """Replays a SAVED real PaddleOCR run from data/ocr_fixtures/<stem>.json.

    This is not live inference and must never be reported as such: the
    OCRResult it returns is stamped engine="paddleocr-fixture", and every
    fixture file carries the provenance needed to judge whether it is still
    meaningful — image sha256, paddleocr/paddlepaddle versions, model names,
    device, and generation timestamp (see run_ocr_fixture_dump.py).

    Purpose is offline regression: downstream code (table structure, routing,
    validator) can be tested against realistic OCR output in ~0s instead of
    ~37s/image, without pretending a model ran. A fixture whose image hash no
    longer matches the image on disk is a hard error, not a silent pass —
    stale fixtures are exactly how a "regression test" quietly stops testing
    the thing it names.
    """

    def __init__(self, fixture_dir: Path | None = None, verify_image_hash: bool = True):
        self.fixture_dir = fixture_dir or OCR_FIXTURE_DIR
        self.verify_image_hash = verify_image_hash

    def recognize(self, image_path: Path) -> OCRResult:
        stem = Path(image_path).stem
        fixture_path = self.fixture_dir / f"{stem}.json"
        if not fixture_path.exists():
            # Loud on purpose, unlike MockOCREngine's soft "no stub authored"
            # result. A missing fixture inside an evaluation silently scores
            # 0.0, which is indistinguishable from "the OCR engine read this
            # page and got nothing" — and that misread cost real time once
            # already: preprocessed images (<name>.processed.png) had no
            # fixtures, so a preprocessing A/B appeared to show a 16-point
            # drop that was pure missing data.
            raise FileNotFoundError(
                f"no OCR fixture for {Path(image_path).name} at {fixture_path}. "
                "Generate it with: python -m src.run_ocr_fixture_dump --images "
                f"{Path(image_path).name}"
            )
        data = json.loads(fixture_path.read_text(encoding="utf-8"))

        meta = data.get("provenance")
        if not meta:
            # A fixture without provenance is indistinguishable from a
            # hand-written stub, which defeats the entire point of this class.
            raise ValueError(f"fixture {fixture_path} has no provenance block")

        if self.verify_image_hash and Path(image_path).exists():
            actual = sha256_file(Path(image_path))
            recorded = meta.get("image_sha256")
            if recorded and actual != recorded:
                raise ValueError(
                    f"fixture {fixture_path.name} was generated from a different image "
                    f"(recorded sha256 {recorded[:12]}..., actual {actual[:12]}...); "
                    "regenerate it with run_ocr_fixture_dump.py"
                )

        result = data["result"]
        return OCRResult(
            text=result.get("text", ""),
            blocks=[OCRBlock(**b) for b in result.get("blocks", [])],
            average_confidence=result.get("average_confidence", 0.0),
            engine="paddleocr-fixture",
            engine_available=True,
            error=result.get("error"),
        )


class PaddleOCREngine(OCREngine):
    """Real adapter — genuinely calls paddleocr. Live inference, not a stub.

    History worth keeping, because the first diagnosis was wrong. An earlier
    round recorded this engine as simply unusable on this machine: weights
    download fine (PP-OCRv6 det/rec, UVDoc, doc/textline orientation), but
    predict() raised

        NotImplementedError: (Unimplemented)
        ConvertPirAttribute2RuntimeAttribute not support
        [pir::ArrayAttribute<pir::DoubleAttribute>]
        (at .../new_executor/instruction/onednn/onednn_instruction.cc:118)

    The path in that message is the actual clue: the failure is in the oneDNN
    (Intel CPU acceleration) execution backend, not in PaddleOCR or the
    models. Initializing with enable_mkldnn=False avoids that backend
    entirely and inference works. Verified on all 11 OCR eval images:
    11/11 succeeded, 79.5% mean gold-field containment vs 64.5% for
    MockOCREngine, at ~37s/image (plus ~11s one-time init) on CPU.

    The cost is real — oneDNN is an accelerator, and turning it off is
    slower. That trade (speed for actually working) is why enable_mkldnn is
    a config flag rather than a hard-coded False, and why configs/
    visual_parser.yaml still defaults ocr.engine to "mock": a 7-minute
    evaluation run is not a reasonable default for CI or for someone
    reproducing this repo.

    Verified environment: paddleocr 3.7.0, paddlepaddle 3.3.1 (CPU build,
    commit 7688495), Python 3.12.10, Windows 11, Intel64 Family 6 Model 186.
    """

    def __init__(self, lang: str = "ch", enable_mkldnn: bool = False):
        self._lang = lang
        self._enable_mkldnn = enable_mkldnn
        self._ocr = None
        self._init_error: Optional[str] = None

    def _ensure_initialized(self) -> None:
        if self._ocr is not None or self._init_error is not None:
            return
        try:
            from paddleocr import PaddleOCR  # heavy import, deferred until actually used
            self._ocr = PaddleOCR(lang=self._lang, enable_mkldnn=self._enable_mkldnn)
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
        except Exception as e:  # noqa: BLE001 - surfaced, never replaced with a fake result
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
                    bbox = poly_to_xyxy(rec_polys[i]) if i < len(rec_polys) else []
                    blocks.append(OCRBlock(text=text, bbox=bbox, confidence=score))
                    texts.append(text)
                    confidences.append(score)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            return OCRResult(
                text="\n".join(texts), blocks=blocks, average_confidence=avg_conf,
                engine="paddleocr", engine_available=True, error=None,
            )
        except Exception as e:  # noqa: BLE001 - result shape differs across paddleocr versions
            return OCRResult(
                text="", average_confidence=0.0, engine="paddleocr",
                engine_available=False, error=f"unexpected paddleocr result shape: {e}",
            )


def get_ocr_engine(engine_name: str, enable_mkldnn: bool = False) -> OCREngine:
    if engine_name == "paddleocr":
        return PaddleOCREngine(enable_mkldnn=enable_mkldnn)
    if engine_name == "fixture":
        return FixtureOCREngine()
    return MockOCREngine()
