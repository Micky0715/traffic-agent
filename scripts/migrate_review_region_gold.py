"""Preview (and only on request, write) a v1 -> v2 gold migration.

DEFAULT IS A DRY RUN. It prints what would change and writes nothing. The
source file is never modified: --write-to must name a NEW path that does not
exist yet, and there is no in-place option at all. A person's annotations are
only ever revised by that person.

What migration does to each v1 record, and nothing else:

  schema_version      region_gold/1 -> region_gold/2          (restamped)
  migrated_from       None -> region_gold/1                   (provenance)
  association_status  stays `unknown`                         (NOT inferred)
  value_legible       stays None                              (NOT inferred)
  candidate_bbox      stays None                              (NOT inferred)
  candidate_fields    stays None                              (NOT inferred)

Why nothing is inferred:
  * association: v1 never asked it. A gold_bbox on a v1 row says where a person
    located the field region; it does not say they checked the value belongs
    to that field. Filling `confirmed` from it would be inventing the answer.
  * value_legible: the v1 guide used `value_visible` for legibility. Copying it
    across would treat that looser annotation as a precise one.
  * candidate_*: would have to come from OCR or the resolver. Neither is gold.

This module reads no OCR output, no resolver output and no VLM cache, and a
test checks its imports to keep it that way.

    python scripts/migrate_review_region_gold.py --gold data/review_region_gold_test.jsonl
    python scripts/migrate_review_region_gold.py --gold <v1 file> \\
        --write-to data/<new name>.jsonl --report-json outputs/<new>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.region_gold import (  # noqa: E402
    SCHEMA_V1, SCHEMA_V2, UNKNOWN, RegionGoldRecord, read_jsonl, sha256_file,
    validate_record, write_jsonl,
)

NOT_INFERRED = {
    "association_status": (
        "v1 never recorded whether the located value belongs to the target "
        "field; left `unknown`. A gold_bbox is not attribution."),
    "value_legible": (
        "v1 used value_visible for legibility; copying it would upgrade a "
        "looser annotation into a precise one. Left None."),
    "candidate_bbox": "would have to come from OCR/resolver output, which is not gold",
    "candidate_fields": "would have to come from OCR/resolver output, which is not gold",
}


class MigrationError(ValueError):
    pass


def migrate_record(record: RegionGoldRecord) -> tuple:
    """Return (new_record, changes). v2 records come back unchanged."""
    if record.schema_version != SCHEMA_V1:
        return record, []
    migrated = replace(record, schema_version=SCHEMA_V2, migrated_from=SCHEMA_V1,
                       association_status=UNKNOWN, value_legible=None,
                       candidate_bbox=None, candidate_fields=None)
    changes = [
        {"field": "schema_version", "from": SCHEMA_V1, "to": SCHEMA_V2,
         "kind": "restamped"},
        {"field": "migrated_from", "from": None, "to": SCHEMA_V1,
         "kind": "provenance"},
    ]
    return migrated, changes


def build_report(source: Path, records: Sequence[RegionGoldRecord]) -> Dict:
    rows, migrated = [], []
    for record in records:
        new, changes = migrate_record(record)
        migrated.append(new)
        problems = validate_record(new)
        rows.append({"record_id": record.record_id, "changes": changes,
                     "valid_after_migration": not problems,
                     "problems_after_migration": problems})
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "dry_run": True,
        "records": len(records),
        "records_restamped": sum(1 for r in rows if r["changes"]),
        # The list the task demands: every field set from anything but a person.
        "inferred_fields": [],
        "defaulted_without_inference": sorted(NOT_INFERRED),
        "not_inferred_reasons": NOT_INFERRED,
        "human_judgements_changed": 0,
        "per_record": rows,
        "_migrated": migrated,
    }


def render_diff(report: Dict) -> str:
    lines = [f"source  {report['source']}",
             f"sha256  {report['source_sha256']}",
             f"records {report['records']}  restamped {report['records_restamped']}",
             f"inferred fields: {report['inferred_fields'] or 'NONE'}",
             f"human judgements changed: {report['human_judgements_changed']}", ""]
    for row in report["per_record"]:
        if not row["changes"]:
            lines.append(f"  = {row['record_id']}  (already v2, unchanged)")
            continue
        lines.append(f"  ~ {row['record_id']}")
        for change in row["changes"]:
            lines.append(f"      {change['field']}: {change['from']!r} -> "
                         f"{change['to']!r}  [{change['kind']}]")
        if not row["valid_after_migration"]:
            lines.append(f"      !! invalid after migration: "
                         f"{row['problems_after_migration']}")
    lines += ["", "not inferred, on purpose:"]
    lines += [f"  {k}: {v}" for k, v in report["not_inferred_reasons"].items()]
    return "\n".join(lines)


def _new_path(path: Path, *, forbid: Sequence[Path]) -> Path:
    resolved = path.resolve()
    if resolved in {p.resolve() for p in forbid}:
        raise MigrationError(f"refusing to write over {path}")
    if resolved.exists():
        raise MigrationError(f"{path} already exists; migration never overwrites")
    return path


def run(gold: Path, *, write_to: Optional[Path] = None,
        report_json: Optional[Path] = None) -> Dict:
    if not gold.exists():
        raise MigrationError(f"gold file not found: {gold}")
    records = read_jsonl(gold)
    if not records:
        raise MigrationError(f"gold file has no records: {gold}")
    report = build_report(gold, records)
    migrated: List[RegionGoldRecord] = report.pop("_migrated")

    if any(not r["valid_after_migration"] for r in report["per_record"]):
        raise MigrationError("migration would produce invalid records; "
                             "nothing written")

    if write_to is not None:
        target = _new_path(write_to, forbid=[gold])
        write_jsonl(migrated, target)
        report["dry_run"] = False
        report["written_to"] = str(target)
        report["written_sha256"] = sha256_file(target)

    # The source must be byte-identical afterwards, whatever else happened.
    if sha256_file(gold) != report["source_sha256"]:
        raise MigrationError("source file changed during migration")

    if report_json is not None:
        target = _new_path(report_json, forbid=[gold] + ([write_to] if write_to else []))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--write-to", default=None,
                        help="NEW file for the migrated records; omit for a dry run")
    parser.add_argument("--report-json", default=None,
                        help="NEW file for the migration report")
    args = parser.parse_args(argv)

    def resolve(value):
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    try:
        report = run(resolve(args.gold), write_to=resolve(args.write_to),
                     report_json=resolve(args.report_json))
    except MigrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(render_diff(report))
    print("\nDRY RUN — nothing written." if report["dry_run"]
          else f"\nwritten to {report['written_to']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
