from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from src.llm_router import _client
from src.models import DrawingExtraction

ROOT = Path(__file__).resolve().parents[1]


def _image_data_uri(image_path: Path) -> str:
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"no JSON object found in VL response: {text[:300]!r}")
    return json.loads(match.group(0))


def extract_drawing(image_path: Path, question: str = "") -> DrawingExtraction:
    """Real call to a Qwen-VL-class model to structure a drawing image.

    prompts/06_drawing_extraction.md defines the schema and the "don't invent
    fields that aren't on the drawing" rule. This gateway doesn't expose the
    exact Qwen2.5-VL checkpoint named on the resume, so VL_MODEL_NAME (default
    qwen-vl-max) is the closest real stand-in for testing the same capability.
    """
    client = _client()
    prompt = (ROOT / "prompts" / "06_drawing_extraction.md").read_text(encoding="utf-8")
    model = os.getenv("VL_MODEL_NAME", "qwen-vl-max")
    user_text = "请抽取这张图纸的结构化字段。" + (f"\n\n另外，用户还想知道：{question}" if question else "")

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": _image_data_uri(image_path)}},
            ],
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception:
        # Some VL endpoints reject response_format on multimodal requests; fall back
        # to plain text and pull the JSON object out of it.
        response = client.chat.completions.create(model=model, temperature=0, messages=messages)

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("VL model returned empty content")
    data = _extract_json(content)
    return DrawingExtraction.model_validate(data)
