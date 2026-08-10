from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = ROOT / "outputs" / ".vlm_cache.json"

PROMPT_VERSION = "v1"


def make_cache_key(image_path: Path, bbox: list, prompt_name: str, model_name: str) -> str:
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()[:16]
    payload = f"{image_hash}|{bbox}|{prompt_name}|{PROMPT_VERSION}|{model_name}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VLMCache:
    """Local JSON-file-backed cache keyed by image content + region + prompt
    + model version, so an unchanged region is never re-sent to the VLM.
    """

    def __init__(self, path: Path | None = None, enabled: bool = True):
        self.path = path or DEFAULT_CACHE_PATH
        self.enabled = enabled
        self._store: dict[str, Any] = {}
        if self.enabled and self.path.exists():
            self._store = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        return self._store.get(key)

    def set(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        self._store[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8")
