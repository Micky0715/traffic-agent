"""Render every PDF page in the source manifest to PNG with pdftoppm.

    python scripts/render_cad_pdf.py \\
        --manifest data/cad_public_corpus/manifests/source_manifest_unreviewed.jsonl \\
        --dpi 300

pdftoppm is the ONLY renderer. If it is missing the command fails, prints
Windows install instructions and records the failure; it does not fall back to
another library, because a different backend produces different pixels.

An existing PNG is verified by hash and never overwritten. Native CAD (DWG,
DXF, DGN) and raster sources are not rendered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cad_corpus.corpus import (  # noqa: E402
    CONVERSION_FAILURES, DUPLICATE_CLUSTERS, PAGE_MANIFEST, SPLIT_CANDIDATES,
    CorpusError, Renderer, RendererUnavailable, build_duplicate_clusters,
    build_split_candidates, load_config, pdftoppm_renderer, read_jsonl,
    render_sources, write_jsonl,
)


def run(manifest: Path, dpi: int, *, renderer: Optional[Renderer] = None,
        repo_root: Path = ROOT) -> dict:
    cfg = load_config()
    if dpi != cfg.render["dpi"]:
        raise CorpusError(f"--dpi {dpi} differs from configs/cad_corpus.yaml "
                          f"({cfg.render['dpi']}); one corpus, one resolution")
    if not manifest.exists():
        raise CorpusError(f"source manifest not found: {manifest}")
    sources = read_jsonl(manifest)
    output_root = manifest.parent.parent
    manifests = manifest.parent

    if renderer is None:
        try:
            renderer = pdftoppm_renderer()
        except RendererUnavailable:
            failures = [{"source_id": s["source_id"], "page_no": None,
                         "reason": "renderer_unavailable", "renderer": "pdftoppm"}
                        for s in sorted(sources, key=lambda r: r["original_path"])
                        if s["source_class"] == "pdf"]
            write_jsonl(failures, manifests / CONVERSION_FAILURES)
            raise

    existing = read_jsonl(manifests / PAGE_MANIFEST)
    pages, failures = render_sources(sources, output_root, cfg, renderer,
                                     repo_root=repo_root, existing_pages=existing)
    write_jsonl(pages, manifests / PAGE_MANIFEST)
    write_jsonl(failures, manifests / CONVERSION_FAILURES)

    clusters = build_duplicate_clusters(sources, pages, cfg)
    (manifests / DUPLICATE_CLUSTERS).write_text(
        json.dumps(clusters, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    write_jsonl(build_split_candidates(sources, clusters, cfg),
                output_root / "splits" / SPLIT_CANDIDATES)
    return {"pages": pages, "failures": failures, "clusters": clusters,
            "renderer": renderer.name, "renderer_version": renderer.version}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dpi", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    try:
        result = run(manifest, args.dpi)
    except RendererUnavailable as exc:
        print(f"ERROR: renderer unavailable.\n{exc}", file=sys.stderr)
        return 3
    except CorpusError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"renderer {result['renderer']} ({result['renderer_version']})")
    print(f"pages {len(result['pages'])}  failures {len(result['failures'])}")
    print(f"near duplicates: {result['clusters']['status']}")
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
