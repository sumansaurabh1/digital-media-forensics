"""CapCheck AI-generated image detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

from backend.detectors.ai_detector.base import AIImageDetector


class CapCheckDetector(AIImageDetector):
    MODULE_NAME = "ai_detector"
    MODEL_NAME = "capcheck/ai-human-generated-image-detection"
    CHECKPOINT_PATH = Path(__file__).resolve().parents[3] / "models" / "capcheck-ntire-hardcase"
    LOCAL_FILES_ONLY = True
    SUPPORTED_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})

    def __init__(self) -> None:
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, image_path: Path) -> dict[str, Any]:
        image_path = Path(image_path)

        if not image_path.is_file():
            return self._error(f"Image file not found: {image_path}")

        if image_path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            return self._error(
                f"Unsupported image format. Supported image types: "
                f"{', '.join(sorted(self.SUPPORTED_SUFFIXES))}"
            )

        try:
            self._load_model()

            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {
                key: value.to(self._device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                logits = self._model(**inputs).logits[0]
                probabilities = torch.softmax(logits, dim=-1)

            human_score, ai_score = self._class_scores(probabilities)

            label = (
                "likely_ai"
                if ai_score >= 0.70
                else "likely_human"
                if ai_score <= 0.30
                else "inconclusive"
            )

            confidence = (
                "high"
                if max(human_score, ai_score) >= 0.85
                else "medium"
                if max(human_score, ai_score) >= 0.70
                else "low"
            )

            return {
                "module": self.MODULE_NAME,
                "model": self.MODEL_NAME,
                "status": "success",
                "label": label,
                "human_score": human_score,
                "ai_score": ai_score,
                "confidence": confidence,
                "device": str(self._device),
                "error": None,
            }

        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            return self._error(str(exc))
        except Exception as exc:
            return self._error(str(exc))

    def _load_model(self) -> None:
        if self._model is None:
            self._processor = AutoImageProcessor.from_pretrained(
                self.CHECKPOINT_PATH,
                local_files_only=self.LOCAL_FILES_ONLY,
            )
            self._model = AutoModelForImageClassification.from_pretrained(
                self.CHECKPOINT_PATH,
                local_files_only=self.LOCAL_FILES_ONLY,
            )
            self._model = self._model.to(self._device).eval()

    def _class_scores(self, probabilities: torch.Tensor) -> tuple[float, float]:
        id2label = self._model.config.id2label
        scores = {
            str(label).lower(): float(probabilities[index])
            for index, label in id2label.items()
        }
        return scores["human"], scores["ai-generated"]

    def _error(self, message: str) -> dict[str, Any]:
        return {
            "module": self.MODULE_NAME,
            "model": self.MODEL_NAME,
            "status": "error",
            "label": None,
            "human_score": None,
            "ai_score": None,
            "confidence": None,
            "device": str(self._device),
            "error": message,
        }