from __future__ import annotations

import re
from typing import Any, Dict, Optional


ASSET_PATTERN = re.compile(
    r"(?:[A-Z]\d{1,3}(?:号)?(?:风机|水泵|泵|扶梯|屏蔽门|信号机|设备)?|"
    r"\d+号(?:风机|水泵|泵|扶梯|屏蔽门|信号机)|"
    r"(?:天河站|体育西路站|广州南站|珠江新城站)[A-Z]?\d{0,3}(?:风机|水泵|泵|扶梯|屏蔽门)?)"
)
REG_PATTERN = re.compile(r"\b(?:JTG|TB|GB|CJJ)\s*[A-Z]?\d+[A-Z]?-\d{4}\b", re.I)
DRAWING_PATTERN = re.compile(r"\b(?:DWG|TUN|ELEC|MEP|FAN)-?[A-Z0-9-]{2,}\b", re.I)
TIME_PATTERNS = [
    re.compile(r"近|最近|过去"),
    re.compile(r"\d+\s*(?:小时|天|周|月|年)"),
    re.compile(r"今天|昨日|昨天|本周|上周|本月|上月|当前|现在"),
    re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"),
]
METRIC_KEYWORDS = ["告警次数", "报警次数", "故障次数", "振动值", "温度", "电流", "完成率", "数量", "多少", "趋势", "平均值"]


def first_match(pattern: re.Pattern[str], text: str) -> Optional[str]:
    m = pattern.search(text)
    return m.group(0) if m else None


CHINESE_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _unique(items):
    return list(dict.fromkeys(items))


def extract_slots(text: str) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}
    regulation = first_match(REG_PATTERN, text)
    assets = _unique(ASSET_PATTERN.findall(text))
    if regulation:
        compact_reg = re.sub(r"[^A-Z0-9]", "", regulation.upper())
        assets = [x for x in assets if re.sub(r"[^A-Z0-9]", "", x.upper()) not in compact_reg]
    # Prefer explicit equipment ids over a station-only match when both appear.
    explicit_assets = [x for x in assets if re.search(r"[A-Z]\d|\d+号", x)]
    selected_assets = explicit_assets or assets
    asset = selected_assets if len(selected_assets) > 1 else (selected_assets[0] if selected_assets else None)
    drawing = first_match(DRAWING_PATTERN, text)
    if asset:
        slots["asset_id"] = asset
    if regulation:
        slots["regulation_code"] = regulation.upper()
    if drawing:
        slots["drawing_no"] = drawing.upper()

    absolute_range = re.search(r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\s*(?:至|到|~|～|-)\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})", text)
    duration = re.search(r"(?:近|最近|过去)?\s*(\d+|[一二两三四五六七八九十])\s*(小时|天|周|月|年)", text)
    if duration:
        raw_num = duration.group(1)
        num = CHINESE_NUM.get(raw_num, raw_num)
        slots["time_range"] = f"最近{num}{duration.group(2)}"
    elif absolute_range:
        slots["time_range"] = f"{absolute_range.group(1)}至{absolute_range.group(2)}"
    else:
        tokens = [token for token in ["今天", "昨天", "昨日", "本周", "上周", "本月", "上月", "当前", "现在"] if token in text]
        tokens = ["昨天" if x == "昨日" else x for x in tokens]
        tokens = _unique(tokens)
        if len(tokens) > 1:
            slots["time_range"] = tokens
        elif tokens:
            slots["time_range"] = tokens[0]

    for metric in METRIC_KEYWORDS:
        if metric in text:
            slots["metric"] = metric
            break

    threshold = re.search(r"(?:超过|高于|大于|低于|小于)\s*(\d+(?:\.\d+)?)\s*([a-zA-Z/·%]+)?", text)
    if threshold:
        slots["threshold"] = threshold.group(1) + (threshold.group(2) or "")
    return slots


def has_time_signal(text: str) -> bool:
    return any(p.search(text) for p in TIME_PATTERNS)
