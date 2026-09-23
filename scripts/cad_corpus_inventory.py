"""Inventory public CAD drawings: discover, hash, register, dedup, split candidates.

    python scripts/cad_corpus_inventory.py --input-dir <explicit download dir> \\
        --output-root data/cad_public_corpus

`--input-dir` is required and never defaults to data/. The scan is
non-recursive unless --recursive is given, and data/ itself is refused as a
recursive root. Known repository material (OCR evaluation images, fixtures,
adversarial images, caches) is excluded by directory name either way.

Reads files; never moves, renames, re-encodes or deletes them. Runs no OCR, no
VLM, no network. Re-running on unchanged input writes byte-identical output,
and a generated file that differs from what is on disk (for instance because a
person edited it) is not replaced without --force-regenerate.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cad_corpus.corpus import (  # noqa: E402
    CONVERSION_FAILURES, DUPLICATE_CLUSTERS, INVENTORY_SCOPE, MANIFEST_DIR,
    PAGE_MANIFEST, RENDER_DIR, SOURCE_MANIFEST, SPLIT_CANDIDATES, CorpusError,
    build_duplicate_clusters, build_source_rows, build_split_candidates, discover,
    dumps_row, load_config, read_jsonl, rel, render_review_doc,
    validate_input_dir, write_guarded,
)

REVIEW_DOC = ROOT / "docs" / "cad_source_metadata_review.md"

README = """# CAD 公开语料（Phase 0）

本目录由 `scripts/cad_corpus_inventory.py` 与 `scripts/render_cad_pdf.py` 生成。

- **原始文件不在这里。** 它们留在用户放置的原位置，manifest 用 `original_path`
  引用，不复制、不移动、不重编码。`raw_pdf/ raw_cad/ raw_raster/` 保留为归档目录，
  Phase 0 没有向其中复制任何文件，以免大文件无意义地存两份。
- `rendered_png_300dpi/`：只由 `pdftoppm` 以 300 DPI 渲染，不使用其他渲染器。
- `manifests/`：所有 `*_unreviewed` 文件中的人工字段（agency、source_url、类别、
  是否含标题栏、split）一律为空或 `unreviewed`，等待人工审核。
- 近似重复只是**候选**，阈值未校准；没有合并或删除任何文件。
- 原生 CAD（DWG / DXF / DGN）只登记，不渲染、不解析。

来源类别：本目录内容为**用户下载的公开图纸**，不是企业数据、不是自撰数据、
不是 OCR fixture，也不是真实推理结果。
"""


def run(input_dir: Optional[Path], output_root: Path, *, recursive: bool = False,
        force: bool = False, review_doc: Path = REVIEW_DOC,
        repo_root: Path = ROOT,
        renderer_available: Optional[bool] = None) -> dict:
    cfg = load_config()
    input_dir = validate_input_dir(input_dir, cfg, recursive=recursive,
                                   repo_root=repo_root)
    files = discover(input_dir, cfg, recursive=recursive)
    if not files:
        raise CorpusError(f"no candidate CAD files in {input_dir}")

    sources = build_source_rows(files, input_dir, cfg, repo_root=repo_root)
    manifests = output_root / MANIFEST_DIR
    pages = read_jsonl(manifests / PAGE_MANIFEST)
    clusters = build_duplicate_clusters(sources, pages, cfg)
    splits = build_split_candidates(sources, clusters, cfg)

    if renderer_available is None:
        renderer_available = shutil.which("pdftoppm") is not None
    if pages:
        render_state = f"{len(pages)} page(s) rendered by pdftoppm"
    elif not renderer_available:
        render_state = "blocked: pdftoppm not installed (no other renderer substituted)"
    else:
        render_state = "pending: run scripts/render_cad_pdf.py"

    scope = {
        "input_dir": rel(input_dir, repo_root),
        "recursive": recursive,
        "excluded_directory_names": cfg.excluded_directory_names,
        "files_registered": len(sources),
        "by_class": {c: sum(1 for s in sources if s["source_class"] == c)
                     for c in cfg.source_classes},
        "note": ("Only files directly under input_dir are registered unless "
                 "recursive is true. Nothing was copied, moved or deleted."),
    }

    for sub in ("raw_pdf", "raw_cad", "raw_raster", RENDER_DIR, MANIFEST_DIR, "splits"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    def jsonl(rows):
        return "".join(dumps_row(r) + "\n" for r in rows).encode("utf-8")

    def js(obj):
        return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n").encode("utf-8")

    written = {
        rel(manifests / SOURCE_MANIFEST, repo_root):
            write_guarded(manifests / SOURCE_MANIFEST, jsonl(sources), force=force),
        rel(manifests / INVENTORY_SCOPE, repo_root):
            write_guarded(manifests / INVENTORY_SCOPE, js(scope), force=force),
        rel(manifests / DUPLICATE_CLUSTERS, repo_root):
            write_guarded(manifests / DUPLICATE_CLUSTERS, js(clusters), force=True),
        rel(output_root / "splits" / SPLIT_CANDIDATES, repo_root):
            write_guarded(output_root / "splits" / SPLIT_CANDIDATES, jsonl(splits),
                          force=True),
        rel(output_root / "README.md", repo_root):
            write_guarded(output_root / "README.md", README.encode("utf-8"), force=force),
        rel(review_doc, repo_root):
            write_guarded(review_doc, render_review_doc(sources, splits, clusters,
                                                        render_state).encode("utf-8"),
                          force=force),
    }
    return {"sources": sources, "clusters": clusters, "splits": splits,
            "scope": scope, "render_state": render_state, "written": written}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default=None,
                        help="REQUIRED. The directory holding the downloaded files.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    args = parser.parse_args(argv)

    def resolve(value):
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    try:
        result = run(resolve(args.input_dir), resolve(args.output_root),
                     recursive=args.recursive, force=args.force_regenerate)
    except CorpusError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    sources = result["sources"]
    print(f"registered {len(sources)} file(s): {result['scope']['by_class']}")
    print(f"pdf pages (provisional, PyMuPDF metadata): "
          f"{sum(s['page_count'] or 0 for s in sources)}")
    print(f"exact duplicate clusters: {len(result['clusters']['exact_file_duplicates'])}")
    print(f"near duplicates: {result['clusters']['status']}")
    print(f"render: {result['render_state']}")
    for path, state in result["written"].items():
        print(f"  {state:<9} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
