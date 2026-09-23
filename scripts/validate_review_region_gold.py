"""Check the gold file before anything is allowed to score against it.

Exit code is non-zero on failure, so this can gate the evaluator.

The prediction-copy check deserves a note. An exact bbox match between gold and
what the resolver produced is not proof of anything — a person who looks at the
right cell and a resolver that found the right cell SHOULD agree. But agreement
to the pixel, repeatedly, is also exactly what tracing over the overlay looks
like, and the two are indistinguishable from the file alone. So it is reported
as a WARNING naming the records, for a human to confirm, and never as a verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.vision.region_gold import (  # noqa: E402
    ASSOCIATION_STATUSES, EXCLUDED, HUMAN_REVIEWED, SCHEMA_V1, UNKNOWN,
    UNREVIEWED, image_hash_matches, read_jsonl, status_summary, validate_record,
)

GOLD = ROOT / "data" / "review_region_gold_unreviewed.jsonl"
PACK_MANIFEST = ROOT / "outputs" / "review_region_annotation_pack" / "manifest.json"


def predicted_bboxes() -> dict:
    """What the resolver produced, for the look-alike check only."""
    if not PACK_MANIFEST.exists():
        return {}
    manifest = json.loads(PACK_MANIFEST.read_text(encoding="utf-8"))
    out: dict = {}
    for page in manifest.get("page_details", []):
        for candidate in page.get("candidates", []):
            for field_name in candidate.get("target_fields", []):
                key = f"{page['document_id']}:p1:{field_name}"
                out.setdefault(key, []).append(
                    [round(float(v), 2) for v in candidate.get("bbox", [])])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(GOLD))
    args = parser.parse_args(argv)
    path = Path(args.gold)
    if not path.exists():
        print(f"gold file not found: {path}")
        return 1

    records = read_jsonl(path)
    if not records:
        print(f"no records in {path}")
        return 1

    errors, warnings = [], []

    seen = Counter(r.record_id for r in records)
    for record_id, count in seen.items():
        if count > 1:
            errors.append(f"record_id {record_id!r} appears {count} times")

    sizes: dict = {}
    for record in records:
        image = ROOT / record.image_path
        if image.exists():
            if record.image_path not in sizes:
                frame = cv2.imread(str(image))
                sizes[record.image_path] = (frame.shape[1], frame.shape[0]) \
                    if frame is not None else None
        else:
            errors.append(f"{record.record_id}: image {record.image_path} missing")

        for problem in validate_record(record, sizes.get(record.image_path)):
            errors.append(f"{record.record_id}: {problem}")

        # A gold bbox belongs to the pixels it was drawn on.
        if record.label_status == HUMAN_REVIEWED and image.exists() \
                and not image_hash_matches(record, ROOT):
            errors.append(
                f"{record.record_id}: image_sha256 does not match "
                f"{record.image_path}; the gold describes different pixels")

    predictions = predicted_bboxes()
    look_alikes = []
    for record in records:
        if record.label_status != HUMAN_REVIEWED or not record.gold_bbox:
            continue
        rounded = [round(float(v), 2) for v in record.gold_bbox]
        if rounded in predictions.get(record.record_id, []):
            look_alikes.append(record.record_id)
    for record in records:
        # A candidate value position copied from a predicted crop is the same
        # risk as a copied gold box, one field over.
        if record.label_status == HUMAN_REVIEWED and record.candidate_bbox:
            rounded = [round(float(v), 2) for v in record.candidate_bbox]
            if rounded in predictions.get(record.record_id, []):
                look_alikes.append(f"{record.record_id} (candidate_bbox)")
    if look_alikes:
        warnings.append(
            f"WARNING: gold bbox exactly equals a predicted bbox for "
            f"{len(look_alikes)} record(s); manual verification is required. "
            f"This is NOT evidence of copying — a correct prediction and a "
            f"correct annotation should agree — but agreement to the pixel is "
            f"also what tracing the overlay looks like. Records: {look_alikes}")

    summary = status_summary(records)
    reviewed = summary.get(HUMAN_REVIEWED, 0)

    print(f"records: {len(records)}   {summary}")
    versions = Counter(r.schema_version for r in records)
    association = Counter(r.effective_association for r in records
                          if r.label_status == HUMAN_REVIEWED)
    print(f"schema versions: {dict(versions)}")
    print("attribution (human_reviewed): "
          + ", ".join(f"{s}={association.get(s, 0)}" for s in ASSOCIATION_STATUSES))
    if versions.get(SCHEMA_V1) and association.get(UNKNOWN):
        print(f"note: {versions[SCHEMA_V1]} v1 record(s) read with attribution "
              "`unknown` — v1 never recorded it. They stay out of any formal "
              "attribution-aware accuracy until a person annotates "
              "association_status.")
    for line in warnings:
        print(line)
    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for line in errors:
            print(f"  {line}")
        return 1

    print("SCHEMA OK")
    if reviewed == 0:
        # Not 0%, not 100%. There is no denominator.
        print("0 human-reviewed records; localization metrics are unavailable.")
    else:
        print(f"{reviewed} human-reviewed record(s); "
              f"localization metrics can be computed on those only "
              f"({summary.get(UNREVIEWED, 0)} unreviewed and "
              f"{summary.get(EXCLUDED, 0)} excluded are outside the denominator).")
        if not association.get("confirmed"):
            print("0 records with association_status=confirmed; legacy region-type "
                  "metrics are available, attribution-aware localization metrics "
                  "are not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
