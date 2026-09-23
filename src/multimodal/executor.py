"""Running — or refusing to run — a vision review.

Three modes that support three different claims, kept apart because they are
routinely conflated:

  disabled       nothing was asked. status=not_invoked.
  cache_replay   a recorded real reply, verified against the current inputs.
                 This is offline regression, NOT inference. Its latency is
                 file I/O and is never reported as model latency.
  live           a real call, this run, costing real money.

`live` needs the config to say so AND the caller to pass allow_live=True, which
the CLI only does for an explicit --allow-live-vlm. One switch would eventually
be left on in a committed file.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from src.multimodal.cache import VisionReviewCache, make_cache_key
from src.multimodal.config import MultimodalConfig
from src.multimodal.crop import resolve_review_crop
from src.multimodal.response_schema import (
    SchemaValidationError, extract_json, validate_response,
)
from src.multimodal.schemas import CropResult, ReviewTarget, VisionReviewResult

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"
PROMPT_FILE = "10_vision_field_review.md"


class LiveCallNotAuthorised(RuntimeError):
    """mode=live without both switches thrown."""


class VisionReviewExecutor:
    """Turns review targets into results, one field at a time, under a budget."""

    def __init__(
        self,
        config: MultimodalConfig,
        *,
        allow_live: bool = False,
        client_factory=None,
    ):
        self.config = config
        self.cfg = config.review
        self.allow_live = allow_live
        self.cache = VisionReviewCache(config.cache.path)
        self._client_factory = client_factory
        self.live_calls_made = 0
        self.model_name = os.getenv("VLM_MODEL_NAME", "qwen-vl-max")

        if self.cfg.mode == "live" and not (self.cfg.allow_live_vlm and allow_live):
            raise LiveCallNotAuthorised(
                "mode=live requires multimodal_review.allow_live_vlm: true in "
                "the config AND --allow-live-vlm on the command line; "
                f"config says {self.cfg.allow_live_vlm}, caller says {allow_live}")

    # -- crop -------------------------------------------------------------

    def crop_for(self, target: ReviewTarget, image_path: Path) -> CropResult:
        return resolve_review_crop(target, image_path, self.cfg,
                                   self.config.cache.crop_dir)

    # -- run --------------------------------------------------------------

    def review(self, target: ReviewTarget, image_path: Path) -> VisionReviewResult:
        crop = self.crop_for(target, image_path)
        return self.review_with_crop(target, crop)

    def review_with_crop(
        self, target: ReviewTarget, crop: CropResult,
    ) -> VisionReviewResult:
        if not crop.available:
            # No reliable region. The whole page is NOT the fallback.
            return VisionReviewResult(
                request_id=target.request_id, status="crop_unavailable",
                inference_mode=self.cfg.mode, error_type=crop.reason)

        if self.cfg.mode == "disabled":
            return VisionReviewResult(
                request_id=target.request_id, status="not_invoked",
                inference_mode="disabled",
                input_image_sha256=crop.source_image_sha256,
                crop_sha256=crop.crop_sha256)

        fields = [target.canonical_field_name]
        key = make_cache_key(
            crop_sha256=crop.crop_sha256 or "", prompt_version=self.cfg.prompt_version,
            model_name=self.model_name, schema_version=self.cfg.schema_version,
            field_names=",".join(sorted(fields)))

        cached = self.cache.get(
            key,
            expected_image_sha256=crop.source_image_sha256 or "",
            expected_crop_sha256=crop.crop_sha256 or "",
            expected_model=self.model_name,
            expected_prompt_version=self.cfg.prompt_version,
            expected_schema_version=self.cfg.schema_version,
            require_image_hash_match=self.cfg.require_image_hash_match,
            require_crop_hash_match=self.cfg.require_crop_hash_match,
        )
        if cached is not None:
            cached.request_id = target.request_id
            return cached

        if self.cfg.mode == "cache_replay":
            # Honest miss. Not an error, and not silently upgraded to a call.
            return VisionReviewResult(
                request_id=target.request_id, status="not_invoked",
                inference_mode="cache_replay",
                error_type="no_cache_entry",
                input_image_sha256=crop.source_image_sha256,
                crop_sha256=crop.crop_sha256)

        if self.live_calls_made >= self.cfg.max_live_calls:
            return VisionReviewResult(
                request_id=target.request_id, status="budget_exhausted",
                inference_mode="live",
                error_type=f"max_live_calls={self.cfg.max_live_calls} reached",
                input_image_sha256=crop.source_image_sha256,
                crop_sha256=crop.crop_sha256)

        result = self._call_live(target, crop, fields)
        if result.status == "success" or result.status == "unreadable":
            self.cache.set(
                key, result,
                input_image_sha256=crop.source_image_sha256 or "",
                crop_sha256=crop.crop_sha256 or "",
                model_name=self.model_name,
                prompt_version=self.cfg.prompt_version,
                schema_version=self.cfg.schema_version,
                crop_path=crop.crop_path or "",
                document_id=target.document_id, field_names=fields)
        return result

    # -- the paid part ----------------------------------------------------

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        from src.vision.qwen_vl_adapter import _vlm_client
        return _vlm_client()

    def _call_live(
        self, target: ReviewTarget, crop: CropResult, fields: Sequence[str],
    ) -> VisionReviewResult:
        from src.drawing_extractor import _image_data_uri

        prompt = (PROMPTS_DIR / PROMPT_FILE).read_text(encoding="utf-8")
        asked = "、".join(fields)
        ocr_note = (f"\n已有 OCR 读数（仅供核对，不是答案）：{target.existing_ocr_value}"
                    if target.existing_ocr_value else "\n已有 OCR 读数：无")
        user_text = (
            f"request_id: {target.request_id}\n"
            f"需要复核的字段：{asked}\n"
            f"复核原因：{'、'.join(target.review_reasons)}\n"
            f"裁剪区域类型：{crop.region_scope}"
            f"{ocr_note}\n\n只输出 JSON。")

        started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        base = dict(
            request_id=target.request_id, inference_mode="live",
            provider=os.getenv("VLM_BASE_URL", "openai-compatible-gateway"),
            model_name=self.model_name,
            model_version=os.getenv("VLM_MODEL_VERSION"),
            prompt_version=self.cfg.prompt_version,
            schema_version=self.cfg.schema_version,
            started_at=started_at,
            input_image_sha256=crop.source_image_sha256,
            crop_sha256=crop.crop_sha256,
        )

        self.live_calls_made += 1
        try:
            client = self._client()
            response = client.chat.completions.create(
                model=self.model_name, temperature=0,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {
                            "url": _image_data_uri(Path(crop.crop_path))}},
                    ]},
                ],
            )
            content = response.choices[0].message.content or ""
            latency = (time.perf_counter() - started) * 1000
        except Exception as exc:  # noqa: BLE001 - reported, never retried blindly
            return VisionReviewResult(
                **base, status="call_failed",
                latency_ms=(time.perf_counter() - started) * 1000,
                error_type=f"{type(exc).__name__}: {exc}")

        try:
            payload = extract_json(content)
            parsed, unreadable = validate_response(payload, list(fields))
        except SchemaValidationError as exc:
            # One repair attempt already happened inside extract_json. No more.
            return VisionReviewResult(
                **base, status="schema_failure", latency_ms=latency,
                error_type=str(exc), raw_response=content[:2000])

        readable_any = any(f.readable and f.raw_value for f in parsed)
        return VisionReviewResult(
            **base,
            status="success" if readable_any else "unreadable",
            latency_ms=latency, fields=parsed,
            unreadable_reasons=unreadable,
            raw_response=content[:2000])


# A request that never reached the cache or the model is not a replay and not
# an inference, whatever mode the run was in. Counting the 9 crop_unavailable
# requests in this repo as cache_replay inflated that number to 11 when exactly
# 2 entries were read.
_NO_LOOKUP = ("crop_unavailable", "budget_exhausted")


def summarize(results: Sequence[VisionReviewResult]) -> dict:
    out = {
        "total_review_requests": len(results),
        "disabled": sum(1 for r in results if r.inference_mode == "disabled"
                        and r.status not in _NO_LOOKUP),
        "cache_replay": sum(1 for r in results if r.inference_mode == "cache_replay"
                            and r.status not in _NO_LOOKUP
                            and r.error_type != "no_cache_entry"),
        "real_inference": sum(1 for r in results if r.inference_mode == "live"
                              and r.status not in _NO_LOOKUP),
        "success": sum(1 for r in results if r.status == "success"),
        "schema_failure": sum(1 for r in results if r.status == "schema_failure"),
        "unreadable": sum(1 for r in results if r.status == "unreadable"),
        "crop_unavailable": sum(1 for r in results if r.status == "crop_unavailable"),
        "call_failed": sum(1 for r in results if r.status == "call_failed"),
        "budget_exhausted": sum(1 for r in results if r.status == "budget_exhausted"),
        "not_invoked": sum(1 for r in results if r.status == "not_invoked"),
    }
    return out
