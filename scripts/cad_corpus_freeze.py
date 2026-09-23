"""Freeze / verify the public CAD corpus, Phase 0.

Writes outputs/cad_corpus_phase0_freeze_manifest.json. No earlier freeze
manifest is read as authority or modified.

    python scripts/cad_corpus_freeze.py --write
    python scripts/cad_corpus_freeze.py --verify

Six groups, because they go stale for different reasons:

  cad_original_sources      the downloaded files, by the paths the manifest uses
  cad_rendered_pages        PNGs from pdftoppm (empty while rendering is blocked,
                            and recorded as blocked rather than omitted)
  cad_manifests             generated manifests and split candidates
  cad_phase0_code           library, CLIs, tests, review doc
  cad_phase0_config         configs/cad_corpus.yaml
  cad_renderer_environment  pdftoppm presence and version, DPI
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cad_corpus.corpus import (  # noqa: E402
    CONVERSION_FAILURES, MANIFEST_DIR, PAGE_MANIFEST, SOURCE_MANIFEST, load_config,
    read_jsonl,
)

CORPUS = ROOT / "data" / "cad_public_corpus"
MANIFEST = ROOT / "outputs" / "cad_corpus_phase0_freeze_manifest.json"

CODE = [
    "src/cad_corpus/__init__.py",
    "src/cad_corpus/corpus.py",
    "scripts/cad_corpus_inventory.py",
    "scripts/render_cad_pdf.py",
    "scripts/cad_corpus_freeze.py",
    "tests/test_cad_corpus_phase0.py",
    "docs/cad_source_metadata_review.md",
]
CONFIG = ["configs/cad_corpus.yaml"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hashes(paths: List[str]) -> Dict[str, str]:
    return {p: sha(ROOT / p) for p in paths if (ROOT / p).exists()}


def group(files: Dict[str, str], **extra) -> Dict:
    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(rel.encode())
        digest.update(files[rel].encode())
    total = sum((ROOT / p).stat().st_size for p in files if (ROOT / p).exists())
    return {"files": files, "file_count": len(files), "total_bytes": total,
            "sha256": digest.hexdigest(), **extra}


def renderer_environment() -> Dict:
    exe = shutil.which("pdftoppm")
    version = None
    if exe:
        probe = subprocess.run([exe, "-v"], capture_output=True, text=True)
        version = next((line for line in (probe.stderr or probe.stdout).splitlines()
                        if "version" in line.lower()), "unknown")
    cfg = load_config()
    return {"renderer": cfg.render["renderer"], "available": bool(exe),
            "version": version, "dpi": cfg.render["dpi"],
            "color_mode": cfg.render["color_mode"],
            "substitute_renderer_used": False}


def snapshot() -> Dict:
    manifests = CORPUS / MANIFEST_DIR
    sources = read_jsonl(manifests / SOURCE_MANIFEST)
    pages = read_jsonl(manifests / PAGE_MANIFEST)
    failures = read_jsonl(manifests / CONVERSION_FAILURES)

    originals = {r["original_path"]: sha(ROOT / r["original_path"]) for r in sources}
    recorded_match = all(r["source_sha256"] == originals[r["original_path"]]
                         for r in sources)
    rendered = {p["rendered_path"]: sha(ROOT / p["rendered_path"]) for p in pages
                if (ROOT / p["rendered_path"]).exists()}
    generated = sorted(
        p.relative_to(ROOT).as_posix()
        for p in list(manifests.glob("*")) + list((CORPUS / "splits").glob("*"))
        + [CORPUS / "README.md"] if p.is_file())

    blocked = any(f.get("reason") == "renderer_unavailable" for f in failures)
    return {
        "stage": "cad_public_corpus/phase0",
        "facts": {
            "source_files": len(sources),
            "by_class": {c: sum(1 for s in sources if s["source_class"] == c)
                         for c in ("pdf", "raster", "native_cad")},
            "pdf_pages_provisional": sum(s["page_count"] or 0 for s in sources),
            "pages_rendered": len(pages),
            "conversion_failures": len(failures),
            "render_state": ("blocked_renderer_unavailable" if blocked and not pages
                             else "rendered" if pages else "pending"),
            "manifest_sha256_matches_files": recorded_match,
            "ocr_runs": 0, "vlm_runs": 0, "network_calls": 0,
            "human_gold_created": False, "accuracy_reported": False,
            "native_cad_parsed": False,
            "original_files_copied_or_moved": False,
        },
        "groups": {
            "cad_original_sources": group(originals),
            "cad_rendered_pages": group(
                rendered,
                status=("empty: rendering blocked, pdftoppm not installed"
                        if not rendered else "rendered")),
            "cad_manifests": group(hashes(generated)),
            "cad_phase0_code": group(hashes(CODE)),
            "cad_phase0_config": group(hashes(CONFIG)),
            "cad_renderer_environment": renderer_environment(),
        },
    }


def write() -> None:
    data = snapshot()
    if not data["facts"]["manifest_sha256_matches_files"]:
        print("REFUSING to write: a source file no longer matches its manifest hash")
        sys.exit(1)
    data["frozen_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, g in data["groups"].items():
        if "files" in g:
            print(f"{name:<26} {g['file_count']:>3} files {g['total_bytes'] / 1e6:>8.1f} MB  "
                  f"{g['sha256'][:16]}…  {g.get('status', '')}")
        else:
            print(f"{name:<26} {g}")
    print(f"\nfacts: {data['facts']}")
    print(f"-> {MANIFEST}")


def verify() -> int:
    if not MANIFEST.exists():
        print("no manifest; run with --write first")
        return 2
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    current = snapshot()
    findings = []
    for name, g in recorded["groups"].items():
        if "files" not in g:
            if g != current["groups"][name]:
                findings.append(f"ENVIRONMENT_CHANGED  {name}: {current['groups'][name]}")
            continue
        before, after = g["files"], current["groups"][name]["files"]
        for rel in sorted(set(before) | set(after)):
            if before.get(rel) != after.get(rel):
                findings.append(f"CHANGED  {name}: {rel}")
    if not findings:
        print("FREEZE OK")
        for name, g in current["groups"].items():
            if "files" in g:
                print(f"  {name:<26} {g['file_count']:>3} files  {g['sha256'][:16]}…")
        return 0
    print("FREEZE VIOLATION")
    for line in findings:
        print(f"  {line}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group_ = parser.add_mutually_exclusive_group(required=True)
    group_.add_argument("--write", action="store_true")
    group_.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.write:
        write()
    else:
        sys.exit(verify())


if __name__ == "__main__":
    main()
