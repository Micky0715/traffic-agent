from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

from src.tools import ASSETS, DRAWINGS
from src.vision.config import VisualFallbackConfig
from src.vision.schemas import DrawingParseResult, VisualEvidence

# Reuse the existing mock ledger (data/mock_assets.json, data/mock_drawings.json)
# rather than inventing a parallel data structure. New synthetic drawings used
# by the eval set are deliberately NOT in this ledger for most cases — that's
# the realistic "not yet catalogued" situation, not a data-loading bug.
_ASSET_LEDGER_IDS = {a["asset_id"] for a in ASSETS} | {d["asset_id"] for d in DRAWINGS}


def _cross_check_device_id(
    vlm_id: str,
    ocr_original_id: Optional[str],
    ocr_processed_id: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Returns (status, warning). status is one of confirmed/suspected/conflict.

    Generalizes the spec's three worked examples from a single OCR reading
    to two (original image OCR, preprocessed image OCR) — the OCR pipeline
    round has both, the earlier VLM-fallback-only round only ever had one.
    OCR==VLM==Ledger -> confirmed; any OCR reading disagrees but VLM==Ledger
    -> suspected (lean VLM, all OCR raw values kept); no OCR agrees and VLM
    not in Ledger -> conflict.
    """
    in_ledger = vlm_id in _ASSET_LEDGER_IDS
    ocr_ids = [x for x in (ocr_original_id, ocr_processed_id) if x]

    if not ocr_ids:
        if in_ledger:
            return "confirmed", None
        return "suspected", f"设备编号 {vlm_id} 台账中未找到匹配，仅有 VLM 单一来源"

    if all(x == vlm_id for x in ocr_ids):
        if in_ledger:
            return "confirmed", None
        return "suspected", f"OCR 与 VLM 一致（{vlm_id}），但台账中未找到匹配"

    disagreeing = [x for x in ocr_ids if x != vlm_id]
    if in_ledger:
        note = "、".join(repr(x) for x in disagreeing)
        extra = ""
        if ocr_original_id and ocr_processed_id and ocr_original_id != ocr_processed_id:
            extra = f"（原图 OCR 与预处理图 OCR 本身也不一致：{ocr_original_id!r} vs {ocr_processed_id!r}）"
        return "suspected", f"OCR 读取为 {note}，VLM 读取为 {vlm_id!r} 且与台账匹配，倾向采用 VLM 结果，OCR 原始值已保留{extra}"

    return "conflict", f"VLM 读取为 {vlm_id!r}，OCR 读取为 {disagreeing}，且台账中均未找到匹配，无法自动判定"


def _check_table_completeness(devices, expected_parameter_names: Optional[List[str]]) -> List[str]:
    """`incomplete` per the spec's worked example: 'VLM 只返回 2 个字段，但
    标准表格预期有 3 个'. Only runs when the caller supplies what a complete
    row should look like — there is no generic per-device-type schema
    registry in this repo, guessing one would be exactly the kind of
    fabrication this module is supposed to prevent elsewhere.
    """
    if not expected_parameter_names:
        return []
    warnings = []
    for device in devices:
        present = {p.name for p in device.parameters}
        missing = [n for n in expected_parameter_names if n not in present]
        if missing:
            warnings.append(f"设备 {device.device_id} 缺少预期字段 {missing}，标准表格预期字段为 {expected_parameter_names}")
    return warnings


def _check_duplicate_devices(devices) -> List[str]:
    counts: Dict[str, int] = {}
    for d in devices:
        counts[d.device_id] = counts.get(d.device_id, 0) + 1
    return [
        f"设备编号 {device_id} 在同一结果中出现了 {count} 次，可能是重复抽取"
        for device_id, count in counts.items() if count > 1
    ]


def _check_parameter_unit(name: str, unit: Optional[str], cfg: VisualFallbackConfig) -> Optional[str]:
    whitelist = cfg.parameter_units.get(name)
    if not whitelist or unit is None:
        return None
    if unit not in whitelist:
        return f"参数 {name} 的单位 {unit!r} 不在允许范围 {whitelist} 内，可能存在量级或单位识别错误"
    return None


def _check_ocr_value_conflict(name: str, vlm_raw_text: str, ocr_raw_text: Optional[str]) -> Optional[str]:
    """A stub OCR reading that disagrees with the VLM reading for the same
    parameter is recorded as a conflict, never silently overwritten.
    """
    if not ocr_raw_text or ocr_raw_text == vlm_raw_text:
        return None
    return f"参数 {name}：OCR 读取为 {ocr_raw_text!r}，VLM 读取为 {vlm_raw_text!r}，两者不一致"


class VisualParseValidator:
    def __init__(self, cfg: VisualFallbackConfig):
        self.cfg = cfg

    def validate(
        self,
        result: DrawingParseResult,
        ocr_device_id_hint: Optional[str] = None,
        ocr_processed_device_id_hint: Optional[str] = None,
        ocr_parameter_hints: Optional[Dict[str, Dict[str, str]]] = None,
        expected_parameter_names: Optional[List[str]] = None,
    ) -> DrawingParseResult:
        """ocr_device_id_hint / ocr_processed_device_id_hint: OCR readings of
        the device ID from the original vs. a preprocessed image — the 2-way
        OCR side of the 4-way (OCR original / OCR processed / VLM / ledger)
        cross-check. ocr_parameter_hints: {device_id: {param_name: ocr_raw_text}}.
        expected_parameter_names: if given, devices missing any of these
        parameter names are flagged `incomplete` rather than silently passing.
        """
        out = deepcopy(result)
        ocr_parameter_hints = ocr_parameter_hints or {}
        warnings: list[str] = []
        statuses: list[str] = ["confirmed"]

        for device in out.devices:
            status, warning = _cross_check_device_id(
                device.device_id, ocr_device_id_hint, ocr_processed_device_id_hint
            )
            statuses.append(status)
            if warning:
                warnings.append(warning)
            out.evidence.append(VisualEvidence(
                field="device_id", value=device.device_id, page=out.page_number,
                bbox=[], region_id="", source="vlm",
                confidence=1.0 if status == "confirmed" else (0.6 if status == "suspected" else 0.2),
            ))

            device_ocr_hints = ocr_parameter_hints.get(device.device_id, {})
            for param in device.parameters:
                unit_warning = _check_parameter_unit(param.name, param.unit, self.cfg)
                if unit_warning:
                    statuses.append("conflict")
                    warnings.append(unit_warning)

                ocr_hint = device_ocr_hints.get(param.name)
                value_warning = _check_ocr_value_conflict(param.name, param.raw_text, ocr_hint)
                if value_warning:
                    statuses.append("conflict")
                    warnings.append(value_warning)

        for relation in out.relations:
            corroborated = (
                relation.source_id in _ASSET_LEDGER_IDS and relation.target_id in _ASSET_LEDGER_IDS
            )
            if corroborated:
                relation.confidence = min(
                    relation.confidence + self.cfg.relation_confidence.corroboration_boost,
                    self.cfg.relation_confidence.max_confidence,
                )
                relation.source = "corroborated"
            else:
                relation.source = "vlm_only"

        duplicate_warnings = _check_duplicate_devices(out.devices)
        if duplicate_warnings:
            statuses.append("suspected")
            warnings.extend(duplicate_warnings)

        completeness_warnings = _check_table_completeness(out.devices, expected_parameter_names)
        if completeness_warnings:
            statuses.append("incomplete")
            warnings.extend(completeness_warnings)

        # Priority: an active factual disagreement (conflict) always wins;
        # missing-but-not-contradictory data (incomplete) is worse than an
        # unverified-but-consistent single source (suspected).
        if "conflict" in statuses:
            final_status = "conflict"
        elif "incomplete" in statuses:
            final_status = "incomplete"
        elif "suspected" in statuses:
            final_status = "suspected"
        else:
            final_status = "confirmed"

        out.validation.status = final_status  # type: ignore[assignment]
        out.validation.warnings = warnings
        return out
