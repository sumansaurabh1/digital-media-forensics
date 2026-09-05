"""Offline evaluation helpers for AI-image detectors."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from backend.detectors.ai_detector.base import AIImageDetector


SUPPORTED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_LABEL_DIRECTORIES = (("human", 0), ("ai", 1))


def discover_labeled_images(dataset_root: str | Path) -> Iterator[tuple[Path, int]]:
    """Yield supported dataset images and their fixed ground-truth labels."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")

    directories: list[tuple[Path, int]] = []
    for name, label in _LABEL_DIRECTORIES:
        directory = root / name
        if not directory.is_dir():
            raise ValueError(f"Dataset directory does not exist: {directory}")
        directories.append((directory, label))

    for directory, label in directories:
        for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                yield path, label


def evaluate_detector(
    detector: AIImageDetector, dataset_root: str | Path, threshold: float = 0.5
) -> dict[str, Any]:
    """Evaluate an AI-image detector against a labeled directory dataset."""
    root = Path(dataset_root)
    ground_truth: list[int] = []
    scores: list[float] = []
    predictions: list[int] = []
    failures: list[dict[str, str]] = []
    total_samples = 0

    for image_path, actual_label in discover_labeled_images(root):
        total_samples += 1
        try:
            result = detector.detect(image_path)
            score = result.get("ai_score")
            if result.get("status") != "success" or score is None:
                raise ValueError(str(result.get("error") or "Detector did not return an AI score."))
            score = float(score)
        except Exception as exc:
            failures.append({"path": str(image_path), "error": str(exc)})
            continue

        ground_truth.append(actual_label)
        scores.append(score)
        predictions.append(int(score >= threshold))

    metrics = _metrics(ground_truth, predictions, scores, threshold)
    return {
        "dataset": {
            "root": str(root),
            "total_samples": total_samples,
            "successful": len(ground_truth),
            "failed": len(failures),
        },
        "metrics": metrics,
        "failures": failures,
    }


def _metrics(
    ground_truth: list[int], predictions: list[int], scores: list[float], threshold: float
) -> dict[str, float | list[list[int]] | None]:
    if not ground_truth:
        return {
            "threshold": threshold,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "confusion_matrix": None,
        }

    roc_auc = float(roc_auc_score(ground_truth, scores)) if len(set(ground_truth)) == 2 else None
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(ground_truth, predictions)),
        "precision": float(precision_score(ground_truth, predictions, zero_division=0)),
        "recall": float(recall_score(ground_truth, predictions, zero_division=0)),
        "f1": float(f1_score(ground_truth, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(ground_truth, predictions, labels=[0, 1]).tolist(),
    }
