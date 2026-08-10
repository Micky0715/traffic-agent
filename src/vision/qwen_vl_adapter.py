from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from src.drawing_extractor import _extract_json, _image_data_uri
from src.vision.base import VisionModel, VisionModelError
from src.vision.cache import VLMCache, make_cache_key
from src.vision.schemas import DeviceEntity, DrawingMetadata, VisualRelation

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"

logger = logging.getLogger(__name__)


def _coerce_parameter_value(raw: dict) -> dict:
    """Real bug hit this session: the VLM occasionally puts a non-numeric
    string (e.g. a device ID like "M-13") in the `value` field despite the
    prompt only asking for that when there's a real number — this happens
    on title-block-style tables that mix numeric and identifier fields in
    the same call. Pydantic's `value: Optional[float]` then raises and, with
    no handling, crashes the entire extraction for one bad field. raw_text
    (an unconstrained string) always preserves what the model actually said,
    so dropping just the unparseable numeric value is strictly better than
    losing the whole device."""
    value = raw.get("value")
    if value is not None:
        try:
            float(value)
        except (TypeError, ValueError):
            raw = {**raw, "value": None}
    return raw


def _vlm_client() -> OpenAI:
    # override=True: .env must win over a stale OPENAI_API_KEY/VLM_API_KEY
    # already sitting in the shell environment (see llm_router._client() for
    # the same fix, hit earlier this session for the same reason).
    load_dotenv(ROOT / ".env", override=True)
    api_key = os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("VLM_API_KEY (or OPENAI_API_KEY) is missing")
    timeout = float(os.getenv("VLM_TIMEOUT", "60"))
    return OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout)


class Qwen25VLAdapter(VisionModel):
    """Real VisionModel implementation for a Qwen2.5-VL-class model.

    This gateway does not expose the exact Qwen2.5-VL checkpoint the resume
    names; VLM_MODEL_NAME defaults to qwen-vl-max, the closest available real
    stand-in (see drawing_extractor.py, validated earlier this session).
    Swapping to a different vendor/model only means writing a new adapter
    class and pointing VLM_MODEL_NAME/VLM_BASE_URL at it — nothing else in
    this module depends on Qwen specifically.
    """

    def __init__(self, cache: VLMCache | None = None):
        self.model = os.getenv("VLM_MODEL_NAME", "qwen-vl-max")
        self.max_retries = int(os.getenv("VLM_MAX_RETRIES", "2"))
        self.cache = cache if cache is not None else VLMCache()

    def _call(self, prompt_file: str, image_path: Path, hint: str, bbox: list) -> dict:
        cache_key = make_cache_key(image_path, bbox, prompt_file, self.model)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info("vlm cache hit: %s / %s", prompt_file, image_path.name)
            return cached

        prompt = (PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")
        user_text = "请抽取这张图纸的结构化字段。" + (f"\n\n另外，用户还想知道：{hint}" if hint else "")
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

        client = _vlm_client()
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                try:
                    response = client.chat.completions.create(
                        model=self.model, temperature=0,
                        response_format={"type": "json_object"}, messages=messages,
                    )
                except Exception:
                    response = client.chat.completions.create(
                        model=self.model, temperature=0, messages=messages,
                    )
                content = response.choices[0].message.content
                if not content:
                    raise VisionModelError("VLM returned empty content")
                data = _extract_json(content)
                logger.info(
                    "vlm call ok: %s / %s (%.0fms, attempt %d)",
                    prompt_file, image_path.name, (time.perf_counter() - started) * 1000, attempt + 1,
                )
                self.cache.set(cache_key, data)
                return data
            except Exception as e:  # noqa: BLE001 - network/parse errors are retried, then raised
                last_err = e
                logger.warning(
                    "vlm call failed: %s / %s (attempt %d/%d): %s",
                    prompt_file, image_path.name, attempt + 1, self.max_retries + 1, e,
                )
        raise VisionModelError(f"VLM call failed after {self.max_retries + 1} attempts: {last_err}") from last_err

    def extract_metadata(self, image_path: Path, hint: str = "") -> DrawingMetadata:
        data = self._call("07_drawing_metadata.md", image_path, hint, bbox=[])
        return DrawingMetadata.model_validate(data)

    def extract_table(self, image_path: Path, hint: str = "") -> List[DeviceEntity]:
        data = self._call("08_drawing_parameter_table.md", image_path, hint, bbox=[])
        devices = data.get("devices", [])
        result = []
        for d in devices:
            params = [_coerce_parameter_value(p) for p in d.get("parameters", [])]
            result.append(DeviceEntity.model_validate({**d, "parameters": params}))
        return result

    def extract_relations(self, image_path: Path, hint: str = "") -> List[VisualRelation]:
        data = self._call("09_drawing_relation.md", image_path, hint, bbox=[])
        relations = data.get("relations", [])
        result = []
        for r in relations:
            result.append(VisualRelation(
                source_id=r["source_id"],
                relation=r["relation"],
                target_id=r["target_id"],
                confidence=float(r.get("confidence", 0.0)),
                source_page=0,
            ))
        return result
