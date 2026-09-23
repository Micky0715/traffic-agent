"""CAD corpus Phase 0: inventory, render bookkeeping, dedup, splits.

All fixtures are tiny files built in tmp_path. The real downloaded PDFs are
never used as test fixtures. PyMuPDF is used only to WRITE a two-page test
PDF; rendering is exercised through a fake renderer that is labelled as such,
because pdftoppm is not installed and no other renderer may stand in for it.

No OCR, no VLM, no network.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from src.cad_corpus import corpus as cc
import scripts.cad_corpus_inventory as inv
import scripts.render_cad_pdf as rnd

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cfg():
    return cc.load_config()


def make_pdf(path: Path, pages: int = 2) -> Path:
    import fitz
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=200, height=120)
        page.insert_text((20, 60), f"TEST PAGE {index + 1}")
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path), deflate=True, garbage=4, no_new_id=True)
    document.close()
    return path


@pytest.fixture
def workspace(tmp_path):
    """A repo-shaped sandbox: downloads/, plus excluded repository material."""
    root = tmp_path / "repo"
    downloads = root / "downloads"
    make_pdf(downloads / "A-sheet.pdf", 2)
    make_pdf(downloads / "B-sheet.pdf", 1)
    (downloads / "C-plan.dwg").write_bytes(b"AC1027 fake dwg header")
    (downloads / "A-sheet.dxf").write_bytes(b"0\nSECTION\n")
    # repository material that must never be registered
    (downloads / "ocr_fixtures").mkdir()
    (downloads / "ocr_fixtures" / "X.png").write_bytes(b"\x89PNG fake")
    (downloads / "drawings").mkdir()
    (downloads / "drawings" / "FAN.png").write_bytes(b"\x89PNG fake")
    return root, downloads


def fake_renderer(counter=None) -> cc.Renderer:
    """TEST ONLY. Writes a deterministic PNG per (pdf, page). It is not
    pdftoppm and is named so that no manifest could mistake it for it."""
    from PIL import Image, ImageDraw

    def render_page(pdf: Path, page_no: int, dpi: int, out: Path) -> None:
        if counter is not None:
            counter.append((pdf.name, page_no))
        image = Image.new("RGB", (120, 80), "white")
        ImageDraw.Draw(image).rectangle([10 * page_no, 10, 60, 50], outline="black")
        image.save(out, format="PNG", optimize=False)

    return cc.Renderer("fake_test_renderer", "test", render_page, None)


def inventory(root, downloads, **kw):
    return inv.run(downloads, root / "corpus", repo_root=root,
                   review_doc=root / "review.md", renderer_available=False, **kw)


# --------------------------------------------------------------------------
# 1  no default scan of data/
# --------------------------------------------------------------------------

def test_1a_input_dir_is_required():
    assert inv.main(["--output-root", "x"]) == 2


def test_1b_data_is_refused_as_a_recursive_root(tmp_path, cfg):
    (tmp_path / "data").mkdir()
    with pytest.raises(cc.CorpusError, match="recursively"):
        cc.validate_input_dir(tmp_path / "data", cfg, recursive=True, repo_root=tmp_path)
    # an explicit non-recursive scan of its top level is allowed
    cc.validate_input_dir(tmp_path / "data", cfg, recursive=False, repo_root=tmp_path)


def test_1c_excluded_repository_material_is_never_registered(workspace):
    root, downloads = workspace
    result = inventory(root, downloads, recursive=True)
    paths = [s["original_path"] for s in result["sources"]]
    assert not any("ocr_fixtures" in p or "drawings" in p for p in paths)


def test_1d_an_input_dir_inside_excluded_material_is_refused(workspace, cfg):
    root, downloads = workspace
    with pytest.raises(cc.CorpusError, match="excluded"):
        cc.validate_input_dir(downloads / "ocr_fixtures", cfg, recursive=False,
                              repo_root=root)


# --------------------------------------------------------------------------
# 2–3  stable ids, byte-identical reruns
# --------------------------------------------------------------------------

def test_2_source_ids_are_stable_and_content_derived(workspace):
    root, downloads = workspace
    first = {s["original_path"]: s["source_id"] for s in inventory(root, downloads)["sources"]}
    second = {s["original_path"]: s["source_id"] for s in inventory(root, downloads)["sources"]}
    assert first == second
    for path, source_id in first.items():
        digest = sha(root / path)
        assert source_id.startswith(f"cad-{digest[:12]}-")


def test_3_rerun_is_byte_identical(workspace):
    root, downloads = workspace
    inventory(root, downloads)
    manifests = root / "corpus" / "manifests"
    before = {p.name: p.read_bytes() for p in manifests.iterdir()}
    doc_before = (root / "review.md").read_bytes()
    result = inventory(root, downloads)
    assert {p.name: p.read_bytes() for p in manifests.iterdir()} == before
    assert (root / "review.md").read_bytes() == doc_before
    assert set(result["written"].values()) == {"unchanged"}


def test_a_hand_edited_manifest_is_not_silently_replaced(workspace):
    root, downloads = workspace
    inventory(root, downloads)
    manifest = root / "corpus" / "manifests" / cc.SOURCE_MANIFEST
    manifest.write_text(manifest.read_text(encoding="utf-8").replace(
        '"agency": null', '"agency": "EDITED"', 1), encoding="utf-8")
    edited = manifest.read_bytes()
    with pytest.raises(cc.CorpusError, match="force-regenerate"):
        inventory(root, downloads)
    assert manifest.read_bytes() == edited


# --------------------------------------------------------------------------
# 4–8  rendering bookkeeping
# --------------------------------------------------------------------------

def test_4_pdf_page_numbers_map_to_png_names(workspace, cfg):
    root, downloads = workspace
    sources = inventory(root, downloads)["sources"]
    pages, failures = cc.render_sources(sources, root / "corpus", cfg,
                                        fake_renderer(), repo_root=root)
    assert failures == []
    a = next(s for s in sources if s["file_name"] == "A-sheet.pdf")
    names = [Path(p["rendered_path"]).name for p in pages
             if p["source_id"] == a["source_id"]]
    assert names == [f"{a['source_id']}__p0001.png", f"{a['source_id']}__p0002.png"]
    assert [p["page_id"] for p in pages if p["source_id"] == a["source_id"]] == \
        [f"{a['source_id']}:p0001", f"{a['source_id']}:p0002"]


def test_5_source_files_are_byte_identical_after_everything(workspace, cfg):
    root, downloads = workspace
    before = {p.name: sha(p) for p in downloads.iterdir() if p.is_file()}
    sources = inventory(root, downloads)["sources"]
    cc.render_sources(sources, root / "corpus", cfg, fake_renderer(), repo_root=root)
    assert {p.name: sha(p) for p in downloads.iterdir() if p.is_file()} == before


def test_6_an_existing_matching_png_is_skipped_and_its_row_kept(workspace, cfg):
    root, downloads = workspace
    sources = inventory(root, downloads)["sources"]
    first, _ = cc.render_sources(sources, root / "corpus", cfg, fake_renderer(),
                                 repo_root=root)
    calls = []
    second, failures = cc.render_sources(sources, root / "corpus", cfg,
                                         fake_renderer(calls), repo_root=root,
                                         existing_pages=first)
    assert calls == [] and failures == []
    assert second == first            # timing fields included: nothing rewritten


def test_7_an_existing_mismatching_png_is_never_overwritten(workspace, cfg):
    root, downloads = workspace
    sources = inventory(root, downloads)["sources"]
    first, _ = cc.render_sources(sources, root / "corpus", cfg, fake_renderer(),
                                 repo_root=root)
    victim = root / first[0]["rendered_path"]
    victim.write_bytes(b"tampered")
    pages, failures = cc.render_sources(sources, root / "corpus", cfg, fake_renderer(),
                                        repo_root=root, existing_pages=first)
    assert victim.read_bytes() == b"tampered"
    assert any(f["reason"] == "existing_png_hash_or_config_mismatch" for f in failures)


def test_7b_an_unrecorded_png_that_differs_is_a_conflict_not_an_overwrite(workspace, cfg):
    root, downloads = workspace
    sources = inventory(root, downloads)["sources"]
    a = next(s for s in sources if s["file_name"] == "A-sheet.pdf")
    stray = root / "corpus" / cc.RENDER_DIR / cc.page_file_name(a["source_id"], 1)
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"not ours")
    _, failures = cc.render_sources(sources, root / "corpus", cfg, fake_renderer(),
                                    repo_root=root)
    assert stray.read_bytes() == b"not ours"
    assert any(f["reason"] == "unrecorded_existing_png_differs" for f in failures)


def test_8_missing_pdftoppm_fails_loudly_with_instructions():
    with pytest.raises(cc.RendererUnavailable, match="winget install"):
        cc.pdftoppm_renderer(which=lambda name: None)


def test_8b_render_cli_records_the_block_and_substitutes_nothing(workspace, monkeypatch):
    root, downloads = workspace
    inventory(root, downloads)
    manifest = root / "corpus" / "manifests" / cc.SOURCE_MANIFEST
    monkeypatch.setattr(rnd, "pdftoppm_renderer",
                        lambda: (_ for _ in ()).throw(cc.RendererUnavailable("x")))
    with pytest.raises(cc.RendererUnavailable):
        rnd.run(manifest, 300, repo_root=root)
    failures = cc.read_jsonl(manifest.parent / cc.CONVERSION_FAILURES)
    assert failures and {f["reason"] for f in failures} == {"renderer_unavailable"}
    assert not list((root / "corpus" / cc.RENDER_DIR).glob("*.png"))


def test_the_renderer_refuses_a_second_resolution(workspace):
    root, downloads = workspace
    inventory(root, downloads)
    with pytest.raises(cc.CorpusError, match="one resolution"):
        rnd.run(root / "corpus" / "manifests" / cc.SOURCE_MANIFEST, 150,
                renderer=fake_renderer(), repo_root=root)


def test_human_judgement_fields_stay_empty_after_rendering(workspace, cfg):
    root, downloads = workspace
    sources = inventory(root, downloads)["sources"]
    pages, _ = cc.render_sources(sources, root / "corpus", cfg, fake_renderer(),
                                 repo_root=root)
    for page in pages:
        assert page["contains_title_block"] is None
        assert page["contains_table"] is None
        assert page["label_status"] == "unreviewed" and page["split"] == "unassigned"


# --------------------------------------------------------------------------
# 9  native CAD
# --------------------------------------------------------------------------

def test_9_native_cad_is_registered_not_rendered(workspace, cfg):
    root, downloads = workspace
    sources = inventory(root, downloads)["sources"]
    cad = [s for s in sources if s["source_class"] == "native_cad"]
    assert {s["extension"] for s in cad} == {".dwg", ".dxf"}
    for row in cad:
        assert row["render_status"] == "not_rendered_native_cad"
        assert row["native_cad_parsing"] == "not_attempted_in_phase0"
        assert row["page_count"] is None
    dxf = next(s for s in cad if s["extension"] == ".dxf")
    assert dxf["same_stem_pdf_present"] is True          # A-sheet.pdf beside it
    pages, _ = cc.render_sources(sources, root / "corpus", cfg, fake_renderer(),
                                 repo_root=root)
    assert not {p["source_id"] for p in pages} & {s["source_id"] for s in cad}


# --------------------------------------------------------------------------
# 10–11  duplicates
# --------------------------------------------------------------------------

def test_10_exact_duplicates_are_clustered_and_all_files_kept(workspace, cfg):
    root, downloads = workspace
    copy = downloads / "A-sheet-copy.pdf"
    copy.write_bytes((downloads / "A-sheet.pdf").read_bytes())
    result = inventory(root, downloads)
    exact = result["clusters"]["exact_file_duplicates"]
    assert len(exact) == 1 and len(exact[0]["members"]) == 2
    assert exact[0]["action"].startswith("none")
    assert copy.exists() and (downloads / "A-sheet.pdf").exists()
    # two paths, two records, two ids
    assert len({*exact[0]["members"]}) == 2


def test_11_near_duplicates_are_only_candidates(workspace, cfg):
    root, downloads = workspace
    sources = inventory(root, downloads)["sources"]
    pages, _ = cc.render_sources(sources, root / "corpus", cfg, fake_renderer(),
                                 repo_root=root)
    clusters = cc.build_duplicate_clusters(sources, pages, cfg)
    near = clusters["near_duplicate_page_candidates"]
    assert near["threshold_status"] == "uncalibrated"
    assert "nothing merged or deleted" in near["note"]
    assert all((root / p["rendered_path"]).exists() for p in pages)


def test_near_duplicate_status_is_explicit_when_nothing_was_rendered(workspace):
    root, downloads = workspace
    clusters = inventory(root, downloads)["clusters"]
    assert clusters["status"] == "near_duplicate_not_computed_no_rendered_pages"


def test_dhash_distance_behaves(tmp_path, cfg):
    from PIL import Image, ImageDraw
    a, b, c = (tmp_path / n for n in ("a.png", "b.png", "c.png"))
    base = Image.new("L", (90, 80), 255)
    ImageDraw.Draw(base).rectangle([10, 10, 50, 60], fill=0)
    base.save(a)
    base.save(b)
    other = Image.new("L", (90, 80), 255)
    ImageDraw.Draw(other).ellipse([40, 5, 88, 75], fill=0)
    other.save(c)
    ha, hb, hc = (cc.dhash(p, 8, 10**8) for p in (a, b, c))
    assert cc.hamming(ha, hb) == 0
    assert cc.hamming(ha, hc) > 0


# --------------------------------------------------------------------------
# 12–13  splits
# --------------------------------------------------------------------------

def test_12_pages_of_one_pdf_never_cross_a_split(workspace, cfg):
    root, downloads = workspace
    sources = inventory(root, downloads)["sources"]
    pages, _ = cc.render_sources(sources, root / "corpus", cfg, fake_renderer(),
                                 repo_root=root)
    splits = cc.build_split_candidates(sources, cc.build_duplicate_clusters(sources, pages, cfg), cfg)
    per_page = cc.page_split(pages, splits)
    by_source = {}
    for page in pages:
        by_source.setdefault(page["source_id"], set()).add(per_page[page["page_id"]])
    assert all(len(values) == 1 for values in by_source.values())


def test_12b_duplicates_share_a_split_group(workspace):
    root, downloads = workspace
    (downloads / "A-sheet-copy.pdf").write_bytes((downloads / "A-sheet.pdf").read_bytes())
    splits = inventory(root, downloads)["splits"]
    groups = {s["original_path"].split("/")[-1]: s["split_group_id"] for s in splits}
    assert groups["A-sheet.pdf"] == groups["A-sheet-copy.pdf"]


def test_13_unknown_agency_is_never_placed_in_gold(workspace):
    root, downloads = workspace
    splits = inventory(root, downloads)["splits"]
    assert all(s["agency"] is None for s in splits)
    assert {s["suggested_split"] for s in splits} == {"unassigned"}
    assert not any("gold" in s["suggested_split"] for s in splits)


def test_agency_is_read_only_from_an_explicit_folder(tmp_path, cfg):
    root = tmp_path / "repo"
    make_pdf(root / "dl" / "TXDOT" / "x.pdf", 1)
    make_pdf(root / "dl" / "y.pdf", 1)
    result = inv.run(root / "dl", root / "corpus", recursive=True, repo_root=root,
                     review_doc=root / "r.md", renderer_available=False)
    rows = {s["file_name"]: s for s in result["sources"]}
    assert rows["x.pdf"]["agency"] == "TXDOT"
    assert rows["x.pdf"]["agency_source"] == "directory_metadata"
    assert rows["y.pdf"]["agency"] is None and rows["y.pdf"]["agency_source"] is None


# --------------------------------------------------------------------------
# 14–18  isolation from everything else in the repo
# --------------------------------------------------------------------------

PHASE0_CODE = [ROOT / "src" / "cad_corpus" / "corpus.py",
               ROOT / "scripts" / "cad_corpus_inventory.py",
               ROOT / "scripts" / "render_cad_pdf.py",
               ROOT / "scripts" / "cad_corpus_freeze.py"]


def _imports(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def test_14_15_no_ocr_vlm_resolver_or_network_imports():
    forbidden = ("ocr", "paddle", "vlm", "qwen", "openai", "multimodal",
                 "resolver", "tables", "rag", "routing", "requests", "urllib",
                 "http", "socket")
    for path in PHASE0_CODE:
        if not path.exists():
            continue
        bad = [m for m in _imports(path) if any(f in m.lower() for f in forbidden)]
        assert not bad, (path.name, bad)


def test_14b_ocr_fixture_directory_is_excluded_by_config(cfg):
    assert "ocr_fixtures" in cfg.excluded_directory_names
    assert "drawings" in cfg.excluded_directory_names
    assert "multimodal_cache" in cfg.excluded_directory_names


@pytest.mark.parametrize("rel_path,expected", [
    ("data/review_region_gold_test.jsonl",
     "dbfc794a9ae188bfdaa9ba8cc68eb6133ebcd0ed1f88f144606feba469583953"),
    ("data/multimodal_cache/vision_review_cache.json",
     "318c5e401c127364e73ad4d44fc1d8be5327cc9259b8cdfced968781d07fef17"),
])
def test_16_existing_gold_and_cache_are_untouched(rel_path, expected):
    assert sha(ROOT / rel_path) == expected


def test_17_18_running_phase0_writes_nothing_outside_its_sandbox(workspace):
    """History and old freeze manifests are hashed before and after a full
    inventory + render cycle in a sandbox."""
    root, downloads = workspace
    watched = sorted((ROOT / "outputs").glob("*freeze_manifest*.json")) + \
        sorted((ROOT / "outputs").glob("review_region_localization_eval*.json"))
    before = {p.name: sha(p) for p in watched}
    sources = inventory(root, downloads)["sources"]
    cc.render_sources(sources, root / "corpus", cc.load_config(), fake_renderer(),
                      repo_root=root)
    assert {p.name: sha(p) for p in watched} == before
