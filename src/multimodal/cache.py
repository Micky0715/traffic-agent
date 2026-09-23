"""The cache of recorded vision-model replies.

Deliberately NOT src/vision/cache.py. That one keys on a hash of
(image, bbox, prompt, model) and stores only the reply, so an entry cannot say
what produced it — you can look up a value but you can never verify it came
from the model, prompt and image you think it did. It also always passed
bbox=[] and the full page, so nothing in it is a crop.

Here every entry carries its own provenance in plain text, and a mismatch
RAISES. A cache that silently falls through to a stale entry when the image
changed is worse than no cache: the report still says cache_replay, and the
number it produced belongs to a different picture.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.multimodal.schemas import VisionReviewResult


class CacheVerificationError(RuntimeError):
    """A cache entry exists but does not match what is being asked for."""


def make_cache_key(*, crop_sha256: str, prompt_version: str,
                   model_name: str, schema_version: str,
                   field_names: str) -> str:
    payload = "|".join([crop_sha256, prompt_version, model_name,
                        schema_version, field_names])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VisionReviewCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._store: Dict[str, Any] = {}
        if self.path.exists():
            self._store = json.loads(self.path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self._store)

    def get(
        self,
        key: str,
        *,
        expected_image_sha256: str,
        expected_crop_sha256: str,
        expected_model: str,
        expected_prompt_version: str,
        expected_schema_version: str,
        require_image_hash_match: bool = True,
        require_crop_hash_match: bool = True,
    ) -> Optional[VisionReviewResult]:
        """Return the recorded reply, or raise if it does not match.

        Returns None only when there is no entry at all. Every other disagreement
        is an error, never a miss: a silent miss would be retried as a live call
        or reported as "no cached evidence", both of which hide the fact that
        the recorded run no longer describes the current inputs.
        """
        entry = self._store.get(key)
        if entry is None:
            return None

        checks = [
            ("schema_version", expected_schema_version, entry.get("schema_version")),
            ("prompt_version", expected_prompt_version, entry.get("prompt_version")),
            ("model_name", expected_model, entry.get("model_name")),
        ]
        if require_crop_hash_match:
            checks.append(("crop_sha256", expected_crop_sha256, entry.get("crop_sha256")))
        if require_image_hash_match:
            checks.append(("input_image_sha256", expected_image_sha256,
                           entry.get("input_image_sha256")))

        for name, expected, actual in checks:
            if expected != actual:
                raise CacheVerificationError(
                    f"cache entry {key[:12]}… {name} mismatch: "
                    f"expected {expected!r}, recorded {actual!r}")

        result = VisionReviewResult.model_validate(entry["result"])
        # The recorded run was live; replaying it is not. Overwritten here so
        # no caller can report a replay as inference.
        result.inference_mode = "cache_replay"
        return result

    def set(self, key: str, result: VisionReviewResult, *,
            input_image_sha256: str, crop_sha256: str,
            model_name: str, prompt_version: str, schema_version: str,
            crop_path: str, document_id: str, field_names: list) -> None:
        self._store[key] = {
            "input_image_sha256": input_image_sha256,
            "crop_sha256": crop_sha256,
            "crop_path": crop_path,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "schema_version": schema_version,
            "document_id": document_id,
            "field_names": field_names,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "recorded_from": "live_inference",
            "result": result.model_dump(mode="json"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8")

    def provenance_summary(self) -> Dict[str, Any]:
        models, prompts = set(), set()
        live = 0
        for entry in self._store.values():
            models.add(entry.get("model_name"))
            prompts.add(entry.get("prompt_version"))
            if entry.get("recorded_from") == "live_inference":
                live += 1
        return {
            "entries": len(self._store),
            "recorded_from_live_inference": live,
            "models": sorted(m for m in models if m),
            "prompt_versions": sorted(p for p in prompts if p),
        }
