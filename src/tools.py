from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.models import SubTask, ToolName, ToolResult


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> List[Dict[str, Any]]:
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


REGULATIONS = _load("mock_regulations.json")
ASSETS = _load("mock_assets.json")
WORK_ORDERS = _load("mock_workorders.json")
DRAWINGS = _load("mock_drawings.json")


def hybrid_search(task: SubTask) -> ToolResult:
    started = time.perf_counter()
    q = task.query.upper()
    hits = []
    for row in REGULATIONS:
        hay = " ".join(str(v) for v in row.values()).upper()
        score = sum(1 for token in ["JTG D81-2017", "应急", "照明", "风机", "振动", "检修", "屏蔽门"] if token in q and token in hay)
        if score > 0:
            hits.append({**row, "score": score})
    hits.sort(key=lambda x: x["score"], reverse=True)
    ok = bool(hits)
    return ToolResult(
        task_id=task.id,
        tool=ToolName.HYBRID_SEARCH,
        ok=ok,
        data=hits[:5],
        error=None if ok else "no_relevant_document",
        latency_ms=int((time.perf_counter() - started) * 1000),
        evidence_score=0.91 if ok else 0.0,
    )


def nl2api_query(task: SubTask) -> ToolResult:
    started = time.perf_counter()
    raw_asset = task.slots.get("asset_id", "")
    asset_ids = raw_asset if isinstance(raw_asset, list) else [raw_asset]
    asset_ids = [str(x) for x in asset_ids if x]
    asset_id = ",".join(asset_ids)
    q = task.query
    if "工单" in q or "维修" in q or "处理记录" in q or "检修记录" in q:
        rows = [x for x in WORK_ORDERS if any(a in x["asset_id"] for a in asset_ids)]
    else:
        rows = [x for x in ASSETS if any(a in x["asset_id"] for a in asset_ids)]

    if rows and any(k in q for k in ["次数", "多少", "统计"]):
        data: Any = {
            "asset_id": asset_id,
            "metric": task.slots.get("metric", "count"),
            "time_range": task.slots.get("time_range"),
            "value": sum(int(x.get("alarm_count_3d", 0)) for x in rows),
            "unit": "次",
            "source": "mock_ops_api",
        }
    else:
        data = rows
    ok = bool(rows)
    return ToolResult(
        task_id=task.id,
        tool=ToolName.NL2API_QUERY,
        ok=ok,
        data=data,
        error=None if ok else "asset_not_found_or_empty_result",
        latency_ms=int((time.perf_counter() - started) * 1000),
        evidence_score=0.95 if ok else 0.0,
    )


def drawing_search(task: SubTask) -> ToolResult:
    started = time.perf_counter()
    drawing_no = str(task.slots.get("drawing_no", ""))
    raw_asset = task.slots.get("asset_id", "")
    asset_ids = raw_asset if isinstance(raw_asset, list) else [raw_asset]
    asset_ids = [str(x) for x in asset_ids if x]
    asset_id = ",".join(asset_ids)
    rows = [
        x for x in DRAWINGS
        if (drawing_no and drawing_no in x["drawing_no"]) or any(a in x["asset_id"] for a in asset_ids)
    ]
    ok = bool(rows)
    return ToolResult(
        task_id=task.id,
        tool=ToolName.DRAWING_SEARCH,
        ok=ok,
        data=rows,
        error=None if ok else "drawing_not_found",
        latency_ms=int((time.perf_counter() - started) * 1000),
        evidence_score=0.9 if ok else 0.0,
    )


def execute_task(task: SubTask) -> ToolResult:
    if task.tool == ToolName.HYBRID_SEARCH:
        return hybrid_search(task)
    if task.tool == ToolName.NL2API_QUERY:
        return nl2api_query(task)
    if task.tool == ToolName.DRAWING_SEARCH:
        return drawing_search(task)
    return ToolResult(task_id=task.id, tool=ToolName.NONE, ok=True, data=None, evidence_score=1.0)
