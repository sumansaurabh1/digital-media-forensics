"""Detector backed by CapCheck's AI-versus-human image classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

from backend.detectors.ai_detector.base import AIImageDetectionResult, AIImageDetector


class CapCheckDetector(AIImageDetector):
    """Classify images with ``capcheck/ai-human-generated-image-detection``.

    The model is loaded once, on the first valid request. Cache-only loading
    prevents an investigation from unexpectedly downloading a replacement.
    Confidence is a simple score band, not scientific certainty.
    """

    MODULE_NAME = "ai_detector"
    MODEL_NAME = "capcheck/ai-human-generated-image-detection"
    CHECKPOINT_PATH = Path(__file__).resolve().parents[3] / "models" / "capcheck-ntire-130k"
    LOCAL_FILES_ONLY = True
    SUPPORTED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor: Any | None = None
        self._model: Any | None = None

    def detect(self, image_path: str | Path) -> AIImageDetectionResult:
        """Return a normalized prediction or a recoverable error result."""
        try:
            path = Path(image_path)
            if not path.is_file():
                return self._result(status="error", error=f"Image path does not exist: {path}")
            if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                return self._result(status="error", error="Supported image types: PNG, JPG, JPEG, WEBP.")

            self._load_model()
            with Image.open(path) as source_image:
                inputs = self._processor(images=source_image.convert("RGB"), return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(self._model(**inputs).logits[0], dim=-1)
            human_score, ai_score = self._class_scores(probabilities.cpu().tolist())
            label = "human" if human_score >= ai_score else "ai"
            top_score = max(human_score, ai_score)
            return self._result(
                status="success", label=label, human_score=human_score, ai_score=ai_score,
                # These are score bands for the model's top class, not scientific certainty.
                confidence="high" if top_score >= 0.9 else "medium" if top_score >= 0.7 else "low",
            )
        except (UnidentifiedImageError, OSError) as exc:
            return self._result(status="error", error=f"Unable to read image: {exc}")
        except Exception as exc:  # Keep detector failures isolated from callers.
            return self._result(status="error", error=f"AI image detection failed: {exc}")

    def _load_model(self) -> None:
        if self._model is None:
            self._processor = AutoImageProcessor.from_pretrained(
                self.CHECKPOINT_PATH, local_files_only=self.LOCAL_FILES_ONLY
            )
            self._model = AutoModelForImageClassification.from_pretrained(
                self.CHECKPOINT_PATH, local_files_only=self.LOCAL_FILES_ONLY
            ).to(self._device).eval()

    def _class_scores(self, probabilities: list[float]) -> tuple[float, float]:
        """Return human and AI scores using the model's configured label names."""
        id2label = self._model.config.id2label
        scores_by_kind: dict[str, float] = {}
        for index, probability in enumerate(probabilities):
            configured_label = id2label.get(index, id2label.get(str(index)))
            if configured_label is None:
                continue
            normalized_label = str(configured_label).lower().replace("_", "-").replace(" ", "-")
            if "human" in normalized_label:
                scores_by_kind["human"] = probability
            elif "ai" in normalized_label or "artificial" in normalized_label:
                scores_by_kind["ai"] = probability

        if set(scores_by_kind) != {"human", "ai"}:
            raise ValueError(f"Unexpected CapCheck labels: {dict(id2label)!r}")
        return scores_by_kind["human"], scores_by_kind["ai"]

    def _result(self, **values: Any) -> AIImageDetectionResult:
        """Build the stable result contract shared with future detector modules."""
        return {
            "module": self.MODULE_NAME, "model": self.MODEL_NAME, "status": "error",
            "label": None, "human_score": None, "ai_score": None, "confidence": None,
            "device": self._device.type, "error": None,
        } | values
