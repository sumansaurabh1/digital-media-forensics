"""Production differential morphing-attack detector."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "morph-detector-v2"
FACE_DETECTOR_PATH = ROOT / "models" / "face" / "yunet.onnx"
FACE_RECOGNIZER_PATH = ROOT / "models" / "face" / "sface.onnx"

_processor = None
_model = None
_face_detector = None
_recognizer = None


class _MorphError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _load():
    global _processor, _model, _face_detector, _recognizer

    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = AutoImageProcessor.from_pretrained(
            MODEL_PATH, local_files_only=True
        )
        _model = (
            AutoModelForImageClassification.from_pretrained(
                MODEL_PATH, local_files_only=True
            )
            .eval()
            .to(device)
        )

    if _face_detector is None:
        _face_detector = cv2.FaceDetectorYN.create(
            str(FACE_DETECTOR_PATH),
            "",
            (320, 320),
            0.2,
            0.3,
            5000,
        )

    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF.create(
            str(FACE_RECOGNIZER_PATH),
            "",
        )

    return _processor, _model, _face_detector, _recognizer


def _face(path: str | Path, role: str):
    try:
        with Image.open(path) as check:
            check.verify()
        image = cv2.imread(str(path))
    except (OSError, UnidentifiedImageError) as exc:
        raise _MorphError(
            "invalid_image",
            f"Unable to read {role}.",
        ) from exc

    if image is None:
        raise _MorphError(
            "invalid_image",
            f"Unable to read {role}.",
        )

    _, _, detector, recognizer = _load()

    height, width = image.shape[:2]
    detector.setInputSize((width, height))

    _, detections = detector.detect(image)

    faces = []
    if detections is not None:
        faces = [
            face
            for face in detections
            if face[2] >= 40
            and face[3] >= 40
            and face[-1] >= 0.20
        ]

    if not faces:
        raise _MorphError(
            "no_face_detected",
            f"No usable face was detected in {role}.",
        )

    face = max(
        faces,
        key=lambda item: float(item[2] * item[3]),
    )

    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned).reshape(-1).astype(np.float32)

    norm = float(np.linalg.norm(feature))
    if not norm:
        raise _MorphError(
            "face_feature_failed",
            f"Could not extract a face feature from {role}.",
        )

    return feature / norm, {
        "bbox": [
            round(float(value), 2)
            for value in face[:4]
        ],
        "confidence": round(float(face[-1]), 6),
        "face_count": len(faces),
        "selected_primary_face": "largest_valid_face",
    }


def _classify(path: str | Path) -> dict[str, Any]:
    processor, model, _, _ = _load()

    with Image.open(path) as image:
        inputs = processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )

    device = next(model.parameters()).device

    with torch.inference_mode():
        logits = model(
            **{
                key: value.to(device)
                for key, value in inputs.items()
            }
        ).logits

    probabilities = torch.softmax(logits, dim=-1)[0]

    labels = {
        int(index): str(name).lower()
        for index, name in model.config.id2label.items()
    }

    morph_index = next(
        (
            index
            for index, label in labels.items()
            if "morph" in label
        ),
        None,
    )

    bona_fide_index = next(
        (
            index
            for index, label in labels.items()
            if "bona" in label
            or "real" in label
            or "human" in label
        ),
        None,
    )

    if morph_index is None or bona_fide_index is None:
        raise _MorphError(
            "morph_model_mapping_error",
            "Morph model has incompatible labels.",
        )

    morph_score = float(probabilities[morph_index])
    bona_fide_score = float(probabilities[bona_fide_index])

    if morph_score >= 0.70:
        label = "morph"
        confidence = (
            "high"
            if morph_score >= 0.85
            else "medium"
        )
    elif morph_score <= 0.30:
        label = "bona_fide"
        confidence = (
            "high"
            if bona_fide_score >= 0.85
            else "medium"
        )
    else:
        label = "inconclusive"
        confidence = "low"

    return {
        "label": label,
        "morph_score": morph_score,
        "bona_fide_score": bona_fide_score,
        "confidence": confidence,
        "model": str(MODEL_PATH),
        "device": str(device),
    }


def analyze_morph(
    target_path: str | Path,
    reference_a_path: str | Path,
    reference_b_path: str | Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "module": "morph_detector",
        "status": "error",
        "label": "inconclusive",
        "confidence": "low",
        "target_face": None,
        "reference_a_face": None,
        "reference_b_face": None,
        "reference_similarities": {},
        "similarity_difference": None,
        "morph_model": None,
        "evidence": [],
    }

    try:
        target, result["target_face"] = _face(
            target_path,
            "target image",
        )

        ref_a, result["reference_a_face"] = _face(
            reference_a_path,
            "reference A",
        )

        model_result = _classify(target_path)

        scores = {
            "target_reference_a": float(
                np.dot(target, ref_a)
            )
        }

        if reference_b_path is not None:
            ref_b, result["reference_b_face"] = _face(
                reference_b_path,
                "reference B",
            )

            scores.update(
                target_reference_b=float(
                    np.dot(target, ref_b)
                ),
                reference_a_reference_b=float(
                    np.dot(ref_a, ref_b)
                ),
            )

            result["similarity_difference"] = abs(
                scores["target_reference_a"]
                - scores["target_reference_b"]
            )

        similarity_a = scores["target_reference_a"]

        if reference_b_path is not None:
            similarity_b = scores["target_reference_b"]

            if (
                similarity_a >= 0.85
                and similarity_b >= 0.85
                and model_result["morph_score"] < 0.70
            ):
                model_result.update(
                    label="bona_fide",
                    confidence="high",
                    decision_reason=(
                        "Both trusted references strongly "
                        "match the target face."
                    ),
                )

        elif (
            similarity_a >= 0.85
            and model_result["morph_score"] < 0.70
        ):
            model_result.update(
                label="bona_fide",
                confidence="high",
                decision_reason=(
                    "Trusted reference strongly matches "
                    "the target face."
                ),
            )

        elif (
            model_result["label"] == "morph"
            and similarity_a < 0.30
        ):
            model_result.update(
                confidence=(
                    "high"
                    if model_result["morph_score"] >= 0.85
                    else "medium"
                ),
                decision_reason=(
                    "Morph classifier agrees with weak "
                    "trusted-reference similarity."
                ),
            )

        result.update(
            status="success",
            label=model_result["label"],
            confidence=model_result["confidence"],
            reference_similarities=scores,
            morph_model=model_result,
            evidence=[
                "Dedicated morph-vs-bona-fide classifier analyzed the target image.",
                "SFace differential face-comparison evidence was calculated.",
            ],
        )

        if reference_b_path is not None:
            result["evidence"].append(
                "Two-reference SFace differential evidence was calculated."
            )

        for key in (
            "target_face",
            "reference_a_face",
            "reference_b_face",
        ):
            if (
                result[key]
                and result[key]["face_count"] > 1
            ):
                result["evidence"].append(
                    f"Multiple faces detected in {key}; "
                    "the largest valid face was selected."
                )

    except _MorphError as exc:
        result.update(
            error_code=exc.code,
            error=str(exc),
        )

    except Exception:
        result.update(
            error_code="analysis_failed",
            error="Morph analysis could not be completed.",
        )

    return result