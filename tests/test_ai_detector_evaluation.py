"""Offline tests for AI-detector evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.detectors.ai_detector.base import AIImageDetectionResult, AIImageDetector
from backend.detectors.ai_detector.evaluation import discover_labeled_images, evaluate_detector


class FakeDetector(AIImageDetector):
    def __init__(self, scores: dict[str, float | None]) -> None:
        self.scores = scores

    def detect(self, image_path: str | Path) -> AIImageDetectionResult:
        score = self.scores[Path(image_path).name]
        if score is None:
            return {
                "module": "fake", "model": "fake", "status": "error", "label": None,
                "human_score": None, "ai_score": None, "confidence": None,
                "device": "cpu", "error": "fake detector failure",
            }
        return {
            "module": "fake", "model": "fake", "status": "success",
            "label": "ai" if score >= 0.5 else "human", "human_score": 1 - score,
            "ai_score": score, "confidence": "high", "device": "cpu", "error": None,
        }


def _image(path: Path) -> Path:
    path.write_bytes(b"test image")
    return path


def _dataset(tmp_path: Path, human: tuple[str, ...] = ("human.jpg",), ai: tuple[str, ...] = ("ai.jpg",)) -> Path:
    root = tmp_path / "dataset"
    for directory, filenames in (("human", human), ("ai", ai)):
        folder = root / directory
        folder.mkdir(parents=True)
        for filename in filenames:
            _image(folder / filename)
    return root


def test_discovery_is_sorted_labeled_and_ignores_unsupported_files(tmp_path: Path) -> None:
    root = _dataset(tmp_path, human=("z.png", "a.jpg"), ai=("b.webp",))
    _image(root / "human" / "ignored.gif")

    found = [(path.name, label) for path, label in discover_labeled_images(root)]

    assert found == [("a.jpg", 0), ("z.png", 0), ("b.webp", 1)]


def test_evaluation_calculates_metrics_and_roc_auc_from_scores(tmp_path: Path) -> None:
    root = _dataset(tmp_path, human=("human-1.jpg", "human-2.jpg"), ai=("ai-1.jpg", "ai-2.jpg"))
    result = evaluate_detector(
        FakeDetector({"human-1.jpg": 0.1, "human-2.jpg": 0.4, "ai-1.jpg": 0.6, "ai-2.jpg": 0.9}), root
    )

    assert result["dataset"] == {"root": str(root), "total_samples": 4, "successful": 4, "failed": 0}
    assert result["metrics"] == {
        "threshold": 0.5, "accuracy": 1.0, "precision": 1.0, "recall": 1.0,
        "f1": 1.0, "roc_auc": 1.0, "confusion_matrix": [[2, 0], [0, 2]],
    }


def test_detector_failure_is_isolated_and_result_is_json_serializable(tmp_path: Path) -> None:
    root = _dataset(tmp_path, human=("human.jpg",), ai=("ai.jpg",))
    result = evaluate_detector(FakeDetector({"human.jpg": 0.1, "ai.jpg": None}), root)

    assert result["dataset"]["successful"] == 1 and result["dataset"]["failed"] == 1
    assert result["failures"] == [{"path": str(root / "ai" / "ai.jpg"), "error": "fake detector failure"}]
    assert result["metrics"]["roc_auc"] is None
    assert result["metrics"]["confusion_matrix"] == [[1, 0], [0, 0]]
    json.dumps(result)


def test_one_class_and_zero_successful_samples_return_safe_metrics(tmp_path: Path) -> None:
    root = _dataset(tmp_path, human=("human.jpg",), ai=())
    one_class = evaluate_detector(FakeDetector({"human.jpg": 0.1}), root)
    assert one_class["metrics"]["roc_auc"] is None
    assert one_class["metrics"]["confusion_matrix"] == [[1, 0], [0, 0]]

    no_successes = evaluate_detector(FakeDetector({"human.jpg": None}), root)
    assert all(value is None for key, value in no_successes["metrics"].items() if key != "threshold")


@pytest.mark.parametrize("path", ("missing", "missing-human", "missing-ai"))
def test_invalid_dataset_structure_has_clear_error(tmp_path: Path, path: str) -> None:
    root = tmp_path / "dataset"
    if path != "missing":
        (root / ("ai" if path == "missing-human" else "human")).mkdir(parents=True)

    with pytest.raises(ValueError, match="Dataset (root|directory) does not exist"):
        list(discover_labeled_images(root))
