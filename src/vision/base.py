from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from src.vision.schemas import DeviceEntity, DrawingMetadata, VisualRelation


class VisionModelError(RuntimeError):
    """Raised when a vision-model call fails after exhausting retries."""


class VisionModel(ABC):
    """Adapter interface so no code outside this module is coupled to a
    specific VLM vendor/version. Swapping models means writing a new adapter
    class, not touching page_router.py / validator.py / chunk_builder.py.
    """

    @abstractmethod
    def extract_metadata(self, image_path: Path, hint: str = "") -> DrawingMetadata:
        ...

    @abstractmethod
    def extract_table(self, image_path: Path, hint: str = "") -> List[DeviceEntity]:
        ...

    @abstractmethod
    def extract_relations(self, image_path: Path, hint: str = "") -> List[VisualRelation]:
        ...
