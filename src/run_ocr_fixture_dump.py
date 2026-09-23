"""Regenerate data/ocr_fixtures/*.json from LIVE PaddleOCR inference.

This is the only script in the repo that runs a real OCR model, and it is
never invoked by the evaluation or the test suite — it must be run by hand:

    python -m src.run_ocr_fixture_dump                 # all eval-set images
    python -m src.run_ocr_fixture_dump --images FAN-A13-02.png
    python -m src.run_ocr_fixture_dump --enable-mkldnn # if oneDNN works here

Every fixture records the provenance needed to decide whether it still means
anything: the sha256 of the exact image bytes, the paddleocr/paddlepaddle
versions, the model names, the device, and when it was generated. A fixture
is evidence of a past real run, NOT a substitute for one — FixtureOCREngine
stamps its results engine="paddleocr-fixture" so no report can quietly
present replayed output as this round's live inference.

Re-run this whenever paddleocr, paddlepaddle, the model set, or a source
image changes. A stale fixture is caught at read time by the image-hash
check, not silently tolerated.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.vision.ocr_engine import OCR_FIXTURE_DIR, PaddleOCREngine, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DRAWINGS_DIR = ROOT / "data" / "drawings"
CASES_PATH = ROOT / "data" / "ocr_pipeline_cases.jsonl"

# The PaddleOCR pipeline instantiates these sub-models; recorded because a
# fixture generated under a different det/rec version is not comparable.
MODEL_SET = [
    "PP-OCRv6_medium_det",
    "PP-OCRv6_medium_rec",
    "PP-LCNet_x1_0_doc_ori",
    "PP-LCNet_x1_0_textline_ori",
    "UVDoc",
]


def _environment(enable_mkldnn: bool) -> dict:
    # Import order is load-bearing on Windows, not stylistic. Importing
    # `paddle` first pulls Paddle's own OpenMP/MKL runtime DLLs into the
    # process; the later `paddleocr` -> paddlex -> modelscope -> torch chain
    # then fails to load torch's DLLs against that already-resident runtime:
    #
    #   OSError: [WinError 127] Error loading "torch\lib\shm.dll"
    #
    # Reproduced deterministically: `import paddleocr` alone succeeds,
    # `import paddle; import paddleocr` does not. paddleocr must come first.
    import paddleocr
    import paddle

    return {
        "paddleocr_version": paddleocr.__version__,
        "paddlepaddle_version": paddle.__version__,
        "paddlepaddle_commit": getattr(paddle.version, "commit", None),
        "models": MODEL_SET,
        "device": paddle.device.get_device(),
        "compiled_with_cuda": paddle.device.is_compiled_with_cuda(),
        "enable_mkldnn": enable_mkldnn,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def _eval_set_images() -> List[str]:
    seen, images = set(), []
    with CASES_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            name = json.loads(line)["image"]
            if name not in seen:
                seen.add(name)
                images.append(name)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real-PaddleOCR fixtures (live inference, slow)")
    parser.add_argument("--images", nargs="*", default=None,
                        help="image file names under data/drawings/ (default: the OCR eval set)")
    parser.add_argument("--enable-mkldnn", action="store_true",
                        help="keep oneDNN on; crashes on machines with the known oneDNN incompatibility")
    parser.add_argument("--out", default=None, help="fixture directory (default: data/ocr_fixtures)")
    args = parser.parse_args()

    images = args.images or _eval_set_images()
    out_dir = Path(args.out) if args.out else OCR_FIXTURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = PaddleOCREngine(enable_mkldnn=args.enable_mkldnn)
    env = _environment(args.enable_mkldnn)
    print(f"paddleocr {env['paddleocr_version']} / paddlepaddle {env['paddlepaddle_version']} "
          f"/ device {env['device']} / enable_mkldnn={args.enable_mkldnn}")
    print(f"generating {len(images)} fixture(s) -> {out_dir}")

    failures = 0
    for name in images:
        image_path = DRAWINGS_DIR / name
        if not image_path.exists():
            print(f"  SKIP {name}: no such image")
            failures += 1
            continue

        started = time.perf_counter()
        result = engine.recognize(image_path)
        elapsed = time.perf_counter() - started

        if not result.engine_available:
            # Do not write a fixture for a failed run — an empty fixture would
            # be indistinguishable from "this image genuinely has no text".
            print(f"  FAIL {name}: {result.error}")
            failures += 1
            continue

        payload = {
            "provenance": {
                "image": name,
                "image_sha256": sha256_file(image_path),
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "generated_by": "src/run_ocr_fixture_dump.py",
                "inference_seconds": round(elapsed, 2),
                **env,
            },
            "result": result.model_dump(mode="json"),
        }
        (out_dir / f"{image_path.stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ok   {name}: {len(result.blocks):>3} blocks, "
              f"conf={result.average_confidence:.3f}, {elapsed:.1f}s")

    print(f"done: {len(images) - failures} written, {failures} failed/skipped")


if __name__ == "__main__":
    main()
