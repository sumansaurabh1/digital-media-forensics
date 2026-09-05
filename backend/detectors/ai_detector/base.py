"""Shared interface and result shape for AI-image detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, TypedDict


class AIImageDetectionResult(TypedDict):
    """The normalized, serializable result returned by an AI-image detector."""

    module: str
    model: str
    status: Literal["success", "error"]
    label: Literal["human", "ai"] | None
    human_score: float | None
    ai_score: float | None
    confidence: Literal["low", "medium", "high"] | None
    device: str
    error: str | None


class AIImageDetector(ABC):
    """Interface implemented by modular AI-image classification detectors."""

    @abstractmethod
    def detect(self, image_path: str | Path) -> AIImageDetectionResult:
        """Classify an image at *image_path* without raising detector errors."""
        raise NotImplementedError
