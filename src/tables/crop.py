"""Save image regions for a table, a row or a cell.

A crop that cannot be traced back to the chunk it illustrates is decoration.
Every saved region records the source image hash, the requested bbox, the bbox
actually cropped after clipping, and the ids of the table/row/cell it belongs
to — so a reviewer can go from an answer to the pixels behind it and back.

File names are derived from document/page/table/row/cell, never from a counter
or a UUID: re-running the pipeline on an unchanged page must overwrite the same
file rather than accumulating copies nobody can match up.

Crops are written only inside the repository's own output directory. Nothing
here copies a file in from elsewhere on the machine.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from src.tables.config import CropConfig
from src.tables.schemas import BBox, ImageRegionRef, ParsedTable, deterministic_id

ROOT = Path(__file__).resolve().parents[2]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clip(bbox: BBox, width: int, height: int,
          padding: int) -> Tuple[List[float], bool]:
    """Pad, then clip to the image. Returns (padded_bbox, clipped_flag).

    Both the requested and the actually-cropped box are kept: near a page edge
    the padding is trimmed, and a reader comparing the crop against a citation
    has to know which of the two they are looking at.
    """
    x0, y0 = bbox[0] - padding, bbox[1] - padding
    x1, y1 = bbox[2] + padding, bbox[3] + padding
    cx0, cy0 = max(0.0, x0), max(0.0, y0)
    cx1, cy1 = min(float(width), x1), min(float(height), y1)
    clipped = (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1)
    return [cx0, cy0, cx1, cy1], clipped


def crop_region(
    image_path: Path,
    bbox: BBox,
    cfg: CropConfig,
    *,
    document_id: str,
    page: int,
    region_type: str,
    table_id: Optional[str] = None,
    row_id: Optional[str] = None,
    cell_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
) -> Optional[ImageRegionRef]:
    """Crop and save one region. None when the region is too small to be useful.

    Returning None rather than a one-pixel image: a degenerate crop still looks
    like evidence in a report while showing nothing.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    height, width = image.shape[:2]

    padded, clipped = _clip(list(bbox), width, height, cfg.padding_px)
    x0, y0, x1, y1 = (int(round(v)) for v in padded)
    if (x1 - x0) < cfg.min_crop_size_px or (y1 - y0) < cfg.min_crop_size_px:
        return None

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    directory = Path(output_dir) if output_dir else ROOT / cfg.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    stem = deterministic_id(document_id, page, region_type,
                            table_id or "", row_id or "", cell_id or "", bbox)
    out_path = directory / f"{document_id}_p{page}_{region_type}_{stem}.png"
    cv2.imwrite(str(out_path), crop)

    try:
        relative = str(out_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        relative = str(out_path)

    return ImageRegionRef(
        image_path=relative,
        source_image_sha256=_sha256_bytes(Path(image_path).read_bytes()),
        crop_sha256=_sha256_bytes(out_path.read_bytes()),
        page=page,
        bbox_original=list(bbox),
        bbox_with_padding=padded,
        region_type=region_type,
        document_id=document_id,
        table_id=table_id, row_id=row_id, cell_id=cell_id,
        clipped=clipped,
    )


def crop_table_region(image_path: Path, table: ParsedTable, cfg: CropConfig,
                      output_dir: Optional[Path] = None) -> Optional[ImageRegionRef]:
    return crop_region(image_path, table.bbox, cfg, document_id=table.document_id,
                       page=table.page_start, region_type="table",
                       table_id=table.table_id, output_dir=output_dir)


def crop_row_region(image_path: Path, table: ParsedTable, row: int,
                    cfg: CropConfig,
                    output_dir: Optional[Path] = None) -> Optional[ImageRegionRef]:
    cells = [table.cell_at(row, col) for col in range(table.col_count)]
    boxes = [c.bbox for c in cells if c and c.bbox]
    if not boxes:
        return None
    bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]
    return crop_region(image_path, bbox, cfg, document_id=table.document_id,
                       page=table.page_start, region_type="row",
                       table_id=table.table_id, row_id=f"r{row}",
                       output_dir=output_dir)


def crop_cell_region(image_path: Path, table: ParsedTable, row: int, col: int,
                     cfg: CropConfig,
                     output_dir: Optional[Path] = None) -> Optional[ImageRegionRef]:
    cell = table.cell_at(row, col)
    if cell is None or not cell.bbox:
        return None
    return crop_region(image_path, cell.bbox, cfg, document_id=table.document_id,
                       page=table.page_start, region_type="cell",
                       table_id=table.table_id, row_id=f"r{row}",
                       cell_id=cell.cell_id, output_dir=output_dir)
