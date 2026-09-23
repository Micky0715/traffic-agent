"""Generate the adversarial stress set.

RULE AUTHOR AND TEST AUTHOR ARE THE SAME PERSON. This set can falsify
assumptions and expose boundaries; it cannot demonstrate generalization.

Design discipline followed here:
  * the six attack categories come from the task spec, not from reading the
    rules, and each one targets a weakness already listed in
    docs/freeze_manifest.md section 5
  * gold is authored from what is DRAWN, never from what the rules accept
  * identifiers deliberately use numbering schemes the rules were never
    induced from (slashes, underscores, dots, CJK), because the drawing-number
    rule's own `origin` says it was generalized from this repo's 11 samples
  * nothing here reads configs/visual_parser.yaml

Output goes to data/adversarial/, kept apart from the regression set.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "adversarial"
GOLD_DIR = OUT_DIR / "gold"

FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
W, H = 760, 760
random.seed(20260915)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def base_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, W - 8, H - 8], outline="black", width=2)
    return img, draw


def draw_title_block(draw: ImageDraw.ImageDraw, rows: list[tuple[str, str]], top: int = 470):
    """Two-column label/value block, the layout the parser expects to read."""
    y = top
    draw.line([30, y - 12, W - 30, y - 12], fill="black", width=1)
    for label, value in rows:
        draw.text((39, y), label, font=font(22), fill="black")
        draw.text((208, y), value, font=font(22), fill="black")
        y += 31
    return y


def draw_header(draw: ImageDraw.ImageDraw, title: str, components: list[str]):
    draw.text((16, 24), title, font=font(28), fill="black")
    x = 48
    for text in components:
        draw.text((x, 97), text, font=font(20), fill="black")
        x += 250


def add_stamp(img: Image.Image, xy: tuple[int, int]):
    draw = ImageDraw.Draw(img)
    draw.ellipse([xy[0], xy[1], xy[0] + 150, xy[1] + 150], outline=(200, 30, 30), width=5)
    draw.text((xy[0] + 26, xy[1] + 62), "审核专用章", font=font(20), fill=(200, 30, 30))


def add_watermark(img: Image.Image, text: str = "严禁外传内部资料"):
    layer = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(layer)
    for row in range(0, H, 90):
        for col in range(-100, W, 260):
            d.text((col + (row // 90) * 40, row), text, font=font(26), fill=(150, 150, 150))
    return Image.blend(img, layer, 0.42)


def degrade(img: Image.Image, kind: str) -> Image.Image:
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=3.4))
    if kind == "skew":
        return img.rotate(-6.5, resample=Image.BICUBIC, fillcolor="white")
    if kind == "low_contrast":
        return Image.blend(img, Image.new("RGB", (W, H), (150, 150, 150)), 0.62)
    return img


def save(img: Image.Image, gold: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / f"{gold['id']}.png")
    (GOLD_DIR / f"{gold['id']}.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")


def single_device_page(case_id, title, drawing_no, rows, components, difficulty,
                       needs_fallback, entities, notes="", visual=None, stamp=False,
                       watermark=False):
    img, draw = base_canvas()
    draw_header(draw, title, components)
    draw_title_block(draw, rows)
    if stamp:
        add_stamp(img, (520, 560))
    if watermark:
        img = add_watermark(img)
    if visual:
        img = degrade(img, visual)
    gold = {
        "id": case_id,
        "drawing_type": "fan_wiring",
        "drawing_no": drawing_no,
        "entities": entities,
        "expected_values": [v for _, v in rows if v],
        # Authored before any pipeline run, from the drawing's own condition:
        # true when the page is visually degraded or deliberately inconsistent
        # enough that one source should not be trusted alone.
        "human_needs_fallback": needs_fallback,
        "difficulty_type": difficulty,
        "notes": notes,
    }
    save(img, gold)
    return gold


def multi_device_page(case_id, title, drawing_no, devices, difficulty, needs_fallback,
                      notes="", visual=None, omit_device_row=None):
    """devices: list of (device_id, power, airflow). omit_device_row drops one
    row's text from the image while keeping it in gold — the drawing says a
    device exists that no reader can see."""
    img, draw = base_canvas()
    draw.text((16, 24), title, font=font(26), fill="black")
    rows = [("图号", drawing_no), ("名称", title)]
    y = draw_title_block(draw, rows, top=170)
    y += 10
    for index, (device_id, power, airflow) in enumerate(devices):
        if omit_device_row is not None and index == omit_device_row:
            y += 93
            continue
        for label, value in ((f"{device_id}设备编号", device_id),
                             (f"{device_id}功率", power),
                             (f"{device_id}风量", airflow)):
            draw.text((39, y), label, font=font(21), fill="black")
            draw.text((250, y), value, font=font(21), fill="black")
            y += 31
    if visual:
        img = degrade(img, visual)
    gold = {
        "id": case_id,
        "drawing_type": "fan_group",
        "drawing_no": drawing_no,
        "entities": [{"entity_type": "fan", "identifier": d,
                      "fields": {"设备编号": d, "功率": p, "风量": a}}
                     for d, p, a in devices],
        "expected_values": [drawing_no] + [d for d, _, _ in devices],
        "human_needs_fallback": needs_fallback,
        "difficulty_type": difficulty,
        "notes": notes,
    }
    save(img, gold)
    return gold


def wiring_entities(motor, breaker, cabinet):
    out = []
    if motor:
        out.append({"entity_type": "motor", "identifier": motor, "fields": {"电机编号": motor}})
    if breaker:
        out.append({"entity_type": "breaker", "identifier": breaker,
                    "fields": {"断路器编号": breaker}})
    if cabinet:
        out.append({"entity_type": "cabinet", "identifier": cabinet,
                    "fields": {"控制柜编号": cabinet}})
    return out


def main() -> None:
    cases = []

    # --- 1. heterogeneous identifier formats (attacks freeze weakness #1: the
    #        drawing-number rule's origin is dataset_induced) ---
    cases.append(single_device_page(
        "ADV01", "B12屏蔽门接线图", "PSD/2026/017",
        [("图号", "PSD/2026/017"), ("名称", "B12屏蔽门接线图"), ("页码", "1/2"),
         ("电机编号", "M/12"), ("断路器编号", "QF/12"), ("控制柜编号", "PSD/CAB/12")],
        ["电机 M/12", "断路器 QF/12"], "heterogeneous_id", False,
        wiring_entities("M/12", "QF/12", "PSD/CAB/12"),
        notes="slash-separated scheme; drawing-number rule expects hyphen segments"))

    cases.append(single_device_page(
        "ADV02", "1号主变压器接线图", "TR_01_A",
        [("图号", "TR_01_A"), ("名称", "1号主变压器接线图"), ("版本", "V2"),
         ("电机编号", "M_01_A"), ("断路器编号", "QF_01_A"), ("控制柜编号", "TR_CAB_01")],
        ["电机 M_01_A", "断路器 QF_01_A"], "heterogeneous_id", False,
        wiring_entities("M_01_A", "QF_01_A", "TR_CAB_01"),
        notes="underscore scheme"))

    cases.append(single_device_page(
        "ADV03", "三号信号机接线图", "SIG.03.2026",
        [("图号", "SIG.03.2026"), ("名称", "三号信号机接线图"), ("页码", "1/1"),
         ("电机编号", "电机三号"), ("断路器编号", "QF.03"), ("控制柜编号", "信号柜三号")],
        ["电机 电机三号", "断路器 QF.03"], "heterogeneous_id", False,
        wiring_entities("电机三号", "QF.03", "信号柜三号"),
        notes="CJK identifiers plus dot separators; charset rule assumes latin codes"))

    # --- 2. unknown equipment types (attacks weakness #7: six mapped type
    #        words) ---
    cases.append(single_device_page(
        "ADV04", "B08屏蔽门控制图", "PSD-B08-03",
        [("图号", "PSD-B08-03"), ("名称", "B08屏蔽门控制图"), ("页码", "2/4"),
         ("电机编号", "M-B08"), ("断路器编号", "QF-B08"), ("控制柜编号", "PSD-CAB-B08")],
        ["屏蔽门 PSD-B08", "驱动器 DRV-B08"], "unknown_type", False,
        wiring_entities("M-B08", "QF-B08", "PSD-CAB-B08"),
        notes="platform screen door: no mapped entity type"))

    cases.append(single_device_page(
        "ADV05", "2号变压器接线图", "TR-02-01",
        [("图号", "TR-02-01"), ("名称", "2号变压器接线图"), ("页码", "1/1"),
         ("电机编号", "M-TR2"), ("断路器编号", "QF-TR2"), ("控制柜编号", "TR-CAB-2")],
        ["变压器 TR-02", "隔离开关 QS-02"], "unknown_type", False,
        wiring_entities("M-TR2", "QF-TR2", "TR-CAB-2"),
        notes="transformer and isolator: unmapped types"))

    cases.append(single_device_page(
        "ADV06", "5号自动扶梯接线图", "ESC-05-01",
        [("图号", "ESC-05-01"), ("名称", "5号自动扶梯接线图"), ("页码", "1/1"),
         ("电机编号", "M-ESC5"), ("断路器编号", "QF-ESC5"), ("控制柜编号", "ESC-CAB-5")],
        ["电扶梯 ESC-05", "制动器 BRK-05"], "unknown_type", False,
        wiring_entities("M-ESC5", "QF-ESC5", "ESC-CAB-5"),
        notes="escalator and brake: unmapped types"))

    # --- 3. entity-count mismatch between sources (attacks the ordinal
    #        mis-pairing risk) ---
    cases.append(multi_device_page(
        "ADV07", "C21/C22/C23风机组接线图", "FAN-MULTI-C2",
        [("C21", "45kW", "28000m3/h"), ("C22", "55kW", "32000m3/h"),
         ("C23", "37kW", "24000m3/h")],
        "entity_count_mismatch", True, omit_device_row=1,
        notes="middle device drawn blank: gold has 3 fans, the page shows 2"))

    cases.append(multi_device_page(
        "ADV08", "D31/D32/D33风机组接线图", "FAN-MULTI-D3",
        [("D31", "45kW", "28000m3/h"), ("D32", "55kW", "32000m3/h"),
         ("D33", "37kW", "24000m3/h")],
        "entity_count_mismatch", True, omit_device_row=2,
        notes="last device drawn blank"))

    cases.append(multi_device_page(
        "ADV09", "E41/E42/E43风机组接线图", "FAN-MULTI-E4",
        [("E43", "37kW", "24000m3/h"), ("E41", "45kW", "28000m3/h"),
         ("E42", "55kW", "32000m3/h")],
        "entity_count_mismatch", False,
        notes="devices drawn out of numeric order; pairing must not use position"))

    # --- 4. near-miss identifiers (attacks the reason entity_ref exists) ---
    cases.append(single_device_page(
        "ADV10", "A19风机接线图", "FAN-A19-07",
        [("图号", "FAN-A19-07"), ("名称", "A19风机接线图"), ("页码", "1/1"),
         ("电机编号", "M-I9"), ("断路器编号", "QF-19"), ("控制柜编号", "FAN-CAB-19")],
        ["电机 M-I9", "断路器 QF-19"], "near_miss_id", True,
        wiring_entities("M-I9", "QF-19", "FAN-CAB-19"),
        notes="capital I where a 1 is expected; sources likely to disagree"))

    cases.append(single_device_page(
        "ADV11", "A23风机接线图", "FAN-A23-07",
        [("图号", "FAN-A23-07"), ("名称", "A23风机接线图"), ("页码", "1/1"),
         ("电机编号", "M-23"), ("断路器编号", "QF-2B"), ("控制柜编号", "FAN-CAB-23")],
        ["电机 M-23", "断路器 QF-2B"], "near_miss_id", True,
        wiring_entities("M-23", "QF-2B", "FAN-CAB-23"),
        notes="QF-2B against the neighbouring QF-23 convention"))

    cases.append(multi_device_page(
        "ADV12", "A16/A1G/A18风机组接线图", "FAN-MULTI-A1",
        [("A16", "45kW", "28000m3/h"), ("A1G", "55kW", "32000m3/h"),
         ("A18", "37kW", "24000m3/h")],
        "near_miss_id", True,
        notes="A1G sits between A16 and A18; G/6 confusion across sources"))

    # --- 5. well-formed but non-existent equipment (attacks the absence of any
    #        ledger check) ---
    cases.append(single_device_page(
        "ADV13", "A77风机接线图", "FAN-A77-01",
        [("图号", "FAN-A77-01"), ("名称", "A77风机接线图"), ("页码", "1/1"),
         ("电机编号", "M-777"), ("断路器编号", "QF-888"), ("控制柜编号", "FAN-CAB-999")],
        ["电机 M-777", "断路器 QF-888"], "format_valid_nonexistent", False,
        wiring_entities("M-777", "QF-888", "FAN-CAB-999"),
        notes="every code passes its shape rule; none of this equipment exists"))

    cases.append(single_device_page(
        "ADV14", "A88风机接线图", "FAN-A88-01",
        [("图号", "FAN-A88-01"), ("名称", "A88风机接线图"), ("页码", "1/1"),
         ("电机编号", "M-0"), ("断路器编号", "QF-00000"), ("控制柜编号", "FAN-CAB-0")],
        ["电机 M-0", "断路器 QF-00000"], "format_valid_nonexistent", False,
        wiring_entities("M-0", "QF-00000", "FAN-CAB-0"),
        notes="degenerate but shape-valid numbering"))

    cases.append(single_device_page(
        "ADV15", "A99风机接线图", "FAN-A99-01",
        [("图号", "FAN-A99-01"), ("名称", "A99风机接线图"), ("页码", "9/1"),
         ("电机编号", "M-99"), ("断路器编号", "QF-99"), ("控制柜编号", "FAN-CAB-99")],
        ["电机 M-99", "断路器 QF-99"], "format_valid_nonexistent", False,
        wiring_entities("M-99", "QF-99", "FAN-CAB-99"),
        notes="page 9 of 1 is impossible but matches the n/m rule"))

    # --- 6. visually hard samples ---
    cases.append(single_device_page(
        "ADV16", "F51风机接线图", "FAN-F51-01",
        [("图号", "FAN-F51-01"), ("名称", "F51风机接线图"), ("页码", "1/1"),
         ("电机编号", "M-F51"), ("断路器编号", "QF-F51"), ("控制柜编号", "FAN-CAB-F51")],
        ["电机 M-F51", "断路器 QF-F51"], "visual_stamp", True,
        wiring_entities("M-F51", "QF-F51", "FAN-CAB-F51"),
        stamp=True, notes="stamp over the cabinet row"))

    cases.append(single_device_page(
        "ADV17", "F52风机接线图", "FAN-F52-01",
        [("图号", "FAN-F52-01"), ("名称", "F52风机接线图"), ("页码", "1/1"),
         ("电机编号", "M-F52"), ("断路器编号", "QF-F52"), ("控制柜编号", "FAN-CAB-F52")],
        ["电机 M-F52", "断路器 QF-F52"], "visual_watermark", True,
        wiring_entities("M-F52", "QF-F52", "FAN-CAB-F52"),
        watermark=True, notes="tiled watermark across the value column"))

    cases.append(single_device_page(
        "ADV18", "F53风机接线图", "FAN-F53-01",
        [("图号", "FAN-F53-01"), ("名称", "F53风机接线图"), ("页码", "1/1"),
         ("电机编号", "M-F53"), ("断路器编号", "QF-F53"), ("控制柜编号", "FAN-CAB-F53")],
        ["电机 M-F53", "断路器 QF-F53"], "visual_blur", True,
        wiring_entities("M-F53", "QF-F53", "FAN-CAB-F53"),
        visual="blur", notes="heavy gaussian blur"))

    cases.append(single_device_page(
        "ADV19", "F54风机接线图", "FAN-F54-01",
        [("图号", "FAN-F54-01"), ("名称", "F54风机接线图"), ("页码", "1/1"),
         ("电机编号", "M-F54"), ("断路器编号", "QF-F54"), ("控制柜编号", "FAN-CAB-F54")],
        ["电机 M-F54", "断路器 QF-F54"], "visual_skew", True,
        wiring_entities("M-F54", "QF-F54", "FAN-CAB-F54"),
        visual="skew", notes="rotated page"))

    cases.append(single_device_page(
        "ADV20", "F55风机接线图", "FAN-F55-01",
        [("图号", "FAN-F55-01"), ("名称", "F55风机接线图"), ("页码", "1/1"),
         ("电机编号", "M-F55"), ("断路器编号", "QF-F55"), ("控制柜编号", "FAN-CAB-F55")],
        ["电机 M-F55", "断路器 QF-F55"], "visual_low_contrast", True,
        wiring_entities("M-F55", "QF-F55", "FAN-CAB-F55"),
        visual="low_contrast", notes="washed-out print"))

    cases.append(single_device_page(
        "ADV21", "F56风机接线图", "FAN-F56-01",
        [("图号", "FAN-F56-01"), ("名称", "F56风机接线图"), ("页码", "1/1"),
         ("电机编号", "M-F56"), ("断路器编号", "QF-F56"), ("控制柜编号", "FAN-CAB-F56")],
        ["电机 M-F56", "断路器 QF-F56"], "visual_stamp_watermark", True,
        wiring_entities("M-F56", "QF-F56", "FAN-CAB-F56"),
        stamp=True, watermark=True, notes="stamp and watermark combined"))

    index = {
        "warning": ("规则作者与压力测试设计者相同，本测试只能用于证伪和发现边界，"
                    "不能证明泛化能力。"),
        "count": len(cases),
        "by_difficulty": {},
        "cases": [{"id": c["id"], "difficulty_type": c["difficulty_type"],
                   "human_needs_fallback": c["human_needs_fallback"],
                   "notes": c["notes"]} for c in cases],
    }
    for c in cases:
        index["by_difficulty"][c["difficulty_type"]] = \
            index["by_difficulty"].get(c["difficulty_type"], 0) + 1
    (OUT_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"generated {len(cases)} adversarial drawings -> {OUT_DIR}")
    for key, value in sorted(index["by_difficulty"].items()):
        print(f"  {key:<28} {value}")


if __name__ == "__main__":
    main()
