"""Public CAD drawing corpus, Phase 0: inventory, deterministic render, dedup.

This phase prepares data and decides nothing about it. No OCR, no VLM, no
resolver, no table recovery, no network. No field rule, device type or title
block template is derived from any page. Every human-judgement field
(agency, source_url, category, contains_title_block, split ...) starts empty
and is left for a person.

Determinism is a requirement, not a nicety: a manifest that changes on every
run cannot be frozen, and a frozen manifest is the only way to prove later
that an evaluation used the pages it says it used. So:

  * ids are derived from content and path, never from time or run order;
  * rows are sorted by relative path; JSON keys are sorted;
  * no wall-clock field is written except render timing, and a page that is
    skipped keeps its original row untouched.

Only ONE renderer is supported: pdftoppm. Rendering with a different library
produces different pixels for the same PDF, and a corpus whose pages came from
two renderers has an invisible confound in every downstream comparison.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "cad_corpus.yaml"

MANIFEST_DIR = "manifests"
SOURCE_MANIFEST = "source_manifest_unreviewed.jsonl"
PAGE_MANIFEST = "page_manifest_unreviewed.jsonl"
DUPLICATE_CLUSTERS = "duplicate_clusters.json"
CONVERSION_FAILURES = "conversion_failures.jsonl"
INVENTORY_SCOPE = "inventory_scope.json"
SPLIT_CANDIDATES = "split_candidates_unreviewed.jsonl"
RENDER_DIR = "rendered_png_300dpi"

WINDOWS_POPPLER_INSTRUCTIONS = """\
pdftoppm (part of Poppler) was not found on PATH. Install it, then re-run.
Windows options (pick one):
  * winget install oschwartz10612.Poppler
  * conda install -c conda-forge poppler
  * download a release from https://github.com/oschwartz10612/poppler-windows/releases,
    unzip it, and add its  Library\\bin  folder to PATH
Check with:  pdftoppm -v
No other renderer is substituted: a different backend renders different pixels."""


class CorpusError(RuntimeError):
    """The run cannot continue without guessing, so it stops."""


class RendererUnavailable(CorpusError):
    pass


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorpusConfig:
    source_classes: Dict[str, List[str]]
    excluded_directory_names: List[str]
    forbidden_recursive_roots: List[str]
    render: Dict
    near_duplicate: Dict
    split: Dict

    def classify(self, extension: str) -> Optional[str]:
        extension = extension.lower()
        for name, extensions in self.source_classes.items():
            if extension in extensions:
                return name
        return None


def load_config(path: Optional[Path] = None) -> CorpusConfig:
    raw = yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    return CorpusConfig(**raw["cad_corpus"])


# ---------------------------------------------------------------------------
# small deterministic helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path, root: Path = ROOT) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def make_source_id(content_sha256: str, relative_path: str) -> str:
    """Stable across runs and machines.

    Content prefix: the same bytes keep the same id wherever the tree is.
    Path suffix: two byte-identical copies at different paths remain two
    records — they are a duplicate CLUSTER, not one row with a lost path.
    """
    path_tag = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:6]
    return f"cad-{content_sha256[:12]}-{path_tag}"


def page_file_name(source_id: str, page_no: int) -> str:
    return f"{source_id}__p{page_no:04d}.png"


def page_id(source_id: str, page_no: int) -> str:
    return f"{source_id}:p{page_no:04d}"


def dumps_row(row: Dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def write_jsonl(rows: Iterable[Dict], path: Path) -> bytes:
    data = "".join(dumps_row(r) + "\n" for r in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_guarded(path: Path, data: bytes, *, force: bool) -> str:
    """Write generated content, but never silently replace a file someone may
    have edited. Identical bytes: no-op. Different bytes: refuse unless forced."""
    if path.exists():
        if path.read_bytes() == data:
            return "unchanged"
        if not force:
            raise CorpusError(
                f"{rel(path)} exists with different content (possibly hand-edited). "
                f"Re-run with --force-regenerate to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return "written"


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def _excluded(path: Path, input_dir: Path, cfg: CorpusConfig) -> bool:
    parts = path.resolve().relative_to(input_dir.resolve()).parts[:-1]
    return any(part in cfg.excluded_directory_names for part in parts)


def validate_input_dir(input_dir: Optional[Path], cfg: CorpusConfig, *,
                       recursive: bool, repo_root: Path = ROOT) -> Path:
    if input_dir is None:
        raise CorpusError("--input-dir is required; there is no default corpus location")
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise CorpusError(f"input directory not found: {input_dir}")
    resolved = input_dir.resolve()
    inside = [p for p in resolved.parts if p in cfg.excluded_directory_names]
    if inside:
        raise CorpusError(f"{input_dir} lies inside excluded repository material {inside}")
    if recursive:
        for name in cfg.forbidden_recursive_roots:
            if resolved == (repo_root / name).resolve():
                raise CorpusError(
                    f"refusing to scan {name}/ recursively: it contains OCR fixtures, "
                    f"evaluation images, adversarial images and caches. Name the "
                    f"download directory, or scan {name}/ non-recursively.")
    return input_dir


def discover(input_dir: Path, cfg: CorpusConfig, *, recursive: bool) -> List[Path]:
    pattern = input_dir.rglob("*") if recursive else input_dir.glob("*")
    found = [p for p in pattern
             if p.is_file() and cfg.classify(p.suffix) is not None
             and not _excluded(p, input_dir, cfg)]
    return sorted(found, key=lambda p: rel(p))


# ---------------------------------------------------------------------------
# source manifest
# ---------------------------------------------------------------------------

def _pdf_page_count(path: Path) -> Tuple[Optional[int], str, Optional[str]]:
    """Page count ONLY. PyMuPDF is used here as a metadata reader and never as
    a renderer; the renderer re-derives the count and the two are compared."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None, "unavailable", "pymupdf not installed"
    try:
        with fitz.open(str(path)) as document:
            return (document.page_count,
                    f"pymupdf {fitz.VersionBind} (metadata only, not a renderer)",
                    None)
    except Exception as exc:  # noqa: BLE001 - a broken PDF is recorded, not fatal
        return None, "pymupdf", f"{type(exc).__name__}: {exc}"


def agency_from_directory(path: Path, input_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """Only an explicit agency folder the user created counts. Files sitting
    directly in the input directory have no directory metadata to read."""
    parts = path.resolve().relative_to(input_dir.resolve()).parts[:-1]
    if not parts:
        return None, None
    return parts[0], "directory_metadata"


def build_source_rows(files: Sequence[Path], input_dir: Path,
                      cfg: CorpusConfig, *, repo_root: Path = ROOT) -> List[Dict]:
    rows = []
    stems_by_class: Dict[str, set] = {}
    for path in files:
        stems_by_class.setdefault(cfg.classify(path.suffix), set()).add(
            (path.parent.resolve(), path.stem.lower()))

    for path in files:
        relative = rel(path, repo_root)
        content = sha256_file(path)
        source_class = cfg.classify(path.suffix)
        agency, agency_source = agency_from_directory(path, input_dir)
        stat = path.stat()

        page_count, count_source, count_error = (None, None, None)
        if source_class == "pdf":
            page_count, count_source, count_error = _pdf_page_count(path)

        row = {
            "source_id": make_source_id(content, relative),
            "original_path": relative,
            "file_name": path.name,
            "extension": path.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "source_sha256": content,
            "source_class": source_class,
            "source_format": path.suffix.lower().lstrip("."),
            "agency": agency,
            "agency_source": agency_source,
            "source_url": None,
            "license_or_terms_url": None,
            "downloaded_at": None,
            # Observed file metadata. NOT a download time and not used as one.
            "file_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime(int(stat.st_mtime))),
            "page_count": page_count,
            "page_count_source": count_source,
            "page_count_status": (
                "provisional_pending_renderer_crosscheck" if page_count is not None
                else ("error" if count_error else "not_applicable")),
            "page_count_error": count_error,
            "render_status": "pending" if source_class == "pdf" else (
                "not_rendered_native_cad" if source_class == "native_cad"
                else "raster_source_not_rendered"),
            "provenance_status": "needs_manual_review",
            "document_category": "unreviewed",
            "used_for_rule_design": False,
            "label_status": "unreviewed",
            "split": "unassigned",
        }
        if source_class == "native_cad":
            parent_stem = (path.parent.resolve(), path.stem.lower())
            row["same_stem_pdf_present"] = parent_stem in stems_by_class.get("pdf", set())
            row["same_stem_raster_present"] = parent_stem in stems_by_class.get("raster", set())
            row["native_cad_parsing"] = "not_attempted_in_phase0"
        rows.append(row)
    return sorted(rows, key=lambda r: r["original_path"])


# ---------------------------------------------------------------------------
# duplicates
# ---------------------------------------------------------------------------

def dhash(image_path: Path, hash_size: int, max_pixels: int) -> str:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = max_pixels
    with Image.open(image_path) as image:
        small = image.convert("L").resize((hash_size + 1, hash_size),
                                          Image.Resampling.LANCZOS)
        # Mode "L": one byte per pixel, row-major — identical to getdata(),
        # which Pillow 14 removes.
        pixels = list(small.tobytes())
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits.append("1" if left > right else "0")
    return f"{int(''.join(bits), 2):0{hash_size * hash_size // 4}x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> List[List[str]]:
        out: Dict[str, List[str]] = {}
        for item in self.parent:
            out.setdefault(self.find(item), []).append(item)
        return [sorted(v) for _, v in sorted(out.items())]


def build_duplicate_clusters(source_rows: Sequence[Dict], page_rows: Sequence[Dict],
                             cfg: CorpusConfig) -> Dict:
    """Candidates for a person. Nothing is merged and nothing is deleted."""
    by_sha: Dict[str, List[Dict]] = {}
    for row in source_rows:
        by_sha.setdefault(row["source_sha256"], []).append(row)
    exact = []
    for digest, rows in sorted(by_sha.items()):
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: r["original_path"])
        exact.append({
            "source_sha256": digest,
            "members": [r["source_id"] for r in rows],
            "paths": [r["original_path"] for r in rows],
            # A suggestion only: shortest path, then lexical. No copy is removed.
            "canonical_candidate": sorted(rows, key=lambda r: (len(r["original_path"]),
                                                               r["original_path"]))[0]["source_id"],
            "action": "none — every file kept",
        })

    nd = cfg.near_duplicate
    hashed = [r for r in page_rows if r.get("dhash")]
    uf = _UnionFind(r["page_id"] for r in hashed)
    pairs = []
    for i, a in enumerate(hashed):
        for b in hashed[i + 1:]:
            distance = hamming(a["dhash"], b["dhash"])
            if distance <= nd["max_hamming_distance"]:
                uf.union(a["page_id"], b["page_id"])
                pairs.append({"a": a["page_id"], "b": b["page_id"], "distance": distance,
                              "cross_source": a["source_id"] != b["source_id"]})
    near = [g for g in uf.groups() if len(g) > 1]
    return {
        "exact_file_duplicates": exact,
        "near_duplicate_page_candidates": {
            "method": nd["method"], "hash_size": nd["hash_size"],
            "max_hamming_distance": nd["max_hamming_distance"],
            "threshold_status": nd["threshold_status"],
            "pages_hashed": len(hashed),
            "clusters": near, "pairs": pairs,
            "note": ("review candidates only; threshold uncalibrated; "
                     "nothing merged or deleted"),
        },
        "status": ("near_duplicate_not_computed_no_rendered_pages" if not hashed
                   else "computed"),
    }


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def build_split_candidates(source_rows: Sequence[Dict], clusters: Dict,
                           cfg: CorpusConfig) -> List[Dict]:
    """One suggestion per SOURCE DOCUMENT, never per page.

    Documents that are exact or near duplicates of each other are grouped
    first, so a copy can never sit on the other side of a split from its
    original. With the agency unknown there is nothing safe to propose: the
    group stays `unassigned` and may never be auto-placed in a gold test.
    """
    ids = [r["source_id"] for r in source_rows]
    uf = _UnionFind(ids)
    for cluster in clusters.get("exact_file_duplicates", []):
        for other in cluster["members"][1:]:
            uf.union(cluster["members"][0], other)
    page_source = {}
    for pair in clusters.get("near_duplicate_page_candidates", {}).get("pairs", []):
        a_src, b_src = pair["a"].split(":p")[0], pair["b"].split(":p")[0]
        if a_src in uf.parent and b_src in uf.parent:
            uf.union(a_src, b_src)
            page_source[a_src] = page_source[b_src] = True

    group_of = {member: f"grp-{index:03d}"
                for index, group in enumerate(uf.groups(), start=1)
                for member in group}
    unknown = cfg.split["unknown_agency_split"]
    out = []
    for row in sorted(source_rows, key=lambda r: r["original_path"]):
        agency = row.get("agency")
        out.append({
            "source_id": row["source_id"],
            "original_path": row["original_path"],
            "split_group_id": group_of[row["source_id"]],
            "split_unit": cfg.split["unit"],
            "agency": agency,
            "suggested_split": unknown if agency is None else "needs_manual_decision",
            "reason": ("agency unknown: cannot place this document in dev or gold "
                       "without risking a shared title block template across the "
                       "split" if agency is None else
                       "agency known: a person decides dev vs gold per agency group"),
            "pages_inherit_document_split": True,
            "status": "unreviewed",
        })
    return out


def page_split(page_rows: Sequence[Dict], split_rows: Sequence[Dict]) -> Dict[str, str]:
    """Every page takes its document's split. There is no per-page choice."""
    by_source = {r["source_id"]: r["suggested_split"] for r in split_rows}
    return {p["page_id"]: by_source[p["source_id"]] for p in page_rows}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

@dataclass
class Renderer:
    name: str
    version: str
    render_page: Callable[[Path, int, int, Path], None]
    page_count: Optional[Callable[[Path], Optional[int]]] = None


def pdftoppm_renderer(which: Callable[[str], Optional[str]] = shutil.which) -> Renderer:
    """The one supported renderer. Missing means stop, never substitute."""
    exe = which("pdftoppm")
    if not exe:
        raise RendererUnavailable(WINDOWS_POPPLER_INSTRUCTIONS)
    probe = subprocess.run([exe, "-v"], capture_output=True, text=True)
    banner = (probe.stderr or probe.stdout or "").strip().splitlines()
    version = next((line for line in banner if "version" in line.lower()), "unknown")

    def render_page(pdf: Path, page_no: int, dpi: int, out_png: Path) -> None:
        prefix = out_png.with_suffix("")
        subprocess.run([exe, "-r", str(dpi), "-png", "-f", str(page_no),
                        "-l", str(page_no), "-singlefile", str(pdf), str(prefix)],
                       check=True, capture_output=True)

    pdfinfo = which("pdfinfo")

    def page_count(pdf: Path) -> Optional[int]:
        if not pdfinfo:
            return None
        info = subprocess.run([pdfinfo, str(pdf)], capture_output=True, text=True)
        match = re.search(r"^Pages:\s+(\d+)", info.stdout, re.M)
        return int(match.group(1)) if match else None

    return Renderer("pdftoppm", version, render_page, page_count)


def render_sources(source_rows: Sequence[Dict], output_root: Path, cfg: CorpusConfig,
                   renderer: Renderer, *, repo_root: Path = ROOT,
                   existing_pages: Optional[Sequence[Dict]] = None) -> Tuple[List[Dict], List[Dict]]:
    """Render every PDF page once. Returns (page_rows, failures).

    An existing PNG is verified, never trusted and never overwritten:
      recorded + same hash  -> skipped, its row kept byte-for-byte
      recorded + different  -> conflict, left alone
      unrecorded            -> rendered to a temp file and compared; adopted
                               only if identical, otherwise a conflict
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = cfg.render["max_image_pixels"]
    dpi = cfg.render["dpi"]
    out_dir = output_root / RENDER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    recorded = {r["page_id"]: r for r in (existing_pages or [])}
    pages, failures = [], []

    for source in sorted(source_rows, key=lambda r: r["original_path"]):
        if source["source_class"] != "pdf":
            continue
        pdf = repo_root / source["original_path"]
        if sha256_file(pdf) != source["source_sha256"]:
            failures.append({"source_id": source["source_id"],
                             "reason": "source_sha256_mismatch", "page_no": None})
            continue
        count = source.get("page_count")
        renderer_count = renderer.page_count(pdf) if renderer.page_count else None
        if renderer_count is not None and count is not None and renderer_count != count:
            failures.append({"source_id": source["source_id"], "page_no": None,
                             "reason": "page_count_disagreement",
                             "manifest_page_count": count,
                             "renderer_page_count": renderer_count})
            continue
        count = renderer_count or count
        if not count:
            failures.append({"source_id": source["source_id"], "page_no": None,
                             "reason": "page_count_unknown"})
            continue

        for page_no in range(1, count + 1):
            pid = page_id(source["source_id"], page_no)
            target = out_dir / page_file_name(source["source_id"], page_no)
            previous = recorded.get(pid)

            if target.exists():
                digest = sha256_file(target)
                if previous and previous.get("rendered_sha256") == digest \
                        and previous.get("source_sha256") == source["source_sha256"] \
                        and previous.get("dpi") == dpi \
                        and previous.get("renderer") == renderer.name:
                    pages.append(previous)          # untouched, incl. its timing
                    continue
                if previous:
                    failures.append({"source_id": source["source_id"], "page_no": page_no,
                                     "reason": "existing_png_hash_or_config_mismatch",
                                     "path": rel(target, repo_root)})
                    continue
                probe = target.with_name(target.stem + ".verify.png")
                renderer.render_page(pdf, page_no, dpi, probe)
                same = probe.exists() and sha256_file(probe) == digest
                probe.unlink(missing_ok=True)
                if not same:
                    failures.append({"source_id": source["source_id"], "page_no": page_no,
                                     "reason": "unrecorded_existing_png_differs",
                                     "path": rel(target, repo_root)})
                    continue
                elapsed = None
            else:
                started = time.perf_counter()
                try:
                    renderer.render_page(pdf, page_no, dpi, target)
                except Exception as exc:  # noqa: BLE001
                    failures.append({"source_id": source["source_id"], "page_no": page_no,
                                     "reason": f"render_error: {type(exc).__name__}"})
                    continue
                elapsed = round((time.perf_counter() - started) * 1000, 1)
                if not target.exists():
                    failures.append({"source_id": source["source_id"], "page_no": page_no,
                                     "reason": "renderer_produced_no_file"})
                    continue

            with Image.open(target) as image:
                width, height = image.size
            pages.append({
                "page_id": pid,
                "source_id": source["source_id"],
                "source_sha256": source["source_sha256"],
                "page_no": page_no,
                "rendered_path": rel(target, repo_root),
                "rendered_sha256": sha256_file(target),
                "width": width, "height": height,
                "dpi": dpi,
                "color_mode": cfg.render["color_mode"],
                "renderer": renderer.name,
                "renderer_version": renderer.version,
                "render_ms": elapsed,
                "render_success": True,
                "dhash": dhash(target, cfg.near_duplicate["hash_size"],
                               cfg.render["max_image_pixels"]),
                "agency": source.get("agency"),
                "document_category": "unreviewed",
                # Deliberately null: a person decides, not a detector.
                "contains_title_block": None,
                "contains_table": None,
                "seen_during_rule_design": False,
                "label_status": "unreviewed",
                "split": "unassigned",
            })
    pages.sort(key=lambda r: (r["source_id"], r["page_no"]))
    failures.sort(key=lambda r: (r["source_id"], r.get("page_no") or 0, r["reason"]))
    return pages, failures


# ---------------------------------------------------------------------------
# review document
# ---------------------------------------------------------------------------

def render_review_doc(source_rows: Sequence[Dict], split_rows: Sequence[Dict],
                      clusters: Dict, render_state: str) -> str:
    split_of = {r["source_id"]: r for r in split_rows}
    lines = [
        "# CAD 公开语料来源信息审核表", "",
        "> 由 `scripts/cad_corpus_inventory.py` 生成。**以下所有待填字段都需要人工填写，"
        "系统没有、也不应该替你猜。** 不要根据 OCR 内容或文件名推断机构。", "",
        f"- 原始文件：**{len(source_rows)}**",
        f"- PDF 总页数（临时，来自 PyMuPDF 元数据）："
        f"**{sum(r['page_count'] or 0 for r in source_rows)}**",
        f"- 渲染状态：**{render_state}**",
        f"- 文件级精确重复组：**{len(clusters['exact_file_duplicates'])}**",
        f"- 近似重复候选：`{clusters['status']}`", "",
        "## 怎么填", "",
        "每个文件一行。填好后告诉我，我再据此冻结最终拆分。",
        "",
        "- **agency**：发布机构（例：某州交通部）。只填你**确定**的，不确定留空。",
        "- **source_url**：你下载它的网页地址。",
        "- **category**：图纸类别（例：标准图、交通信号、排水详图），你自己的判断。",
        "- **是否公开**：下载页面是否公开可访问、条款是否允许研究使用。",
        "- **已用于规则设计**：你或我是否已经**看过这份图并据此改过规则**。"
        "看过的只能进开发集，不能进 Gold Test。",
        "- **建议 split**：`dev` / `gold_test` / `exclude`。"
        "**同一机构、同一标题栏模板的文件必须进同一边。**",
        "- **接受建议**：是 / 否。", "",
        "| # | source_id | 文件名 | 页数 | 大小 MB | 拆分组 | agency | source_url | category | 是否公开 | 已用于规则设计 | 建议 split | 接受建议 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(source_rows, start=1):
        lines.append(
            f"| {index} | `{row['source_id']}` | {row['file_name']} | "
            f"{row['page_count'] if row['page_count'] is not None else '—'} | "
            f"{row['file_size_bytes'] / 1e6:.1f} | "
            f"{split_of[row['source_id']]['split_group_id']} | "
            f"{row['agency'] or '____'} | ____ | ____ | ____ | ____ | ____ | ____ |")
    lines += ["", "## 为什么不能按页面随机拆分", "",
              "同一份 PDF 的所有页共用同一个标题栏、同一个修订表、同一套绘图习惯。"
              "如果把第 3 页放进开发集、第 4 页放进 Gold Test，模型（或规则）"
              "在开发集上学到的其实是**这份图纸的模板**，到了 Gold Test 上"
              "它认的是\"见过的模板\"，测出来的是记忆而不是泛化。"
              "所以拆分单位是**文档**，已知机构后再上升到**机构 / 模板**。", ""]
    return "\n".join(lines) + "\n"
