"""Unit tests for the isolated CapCheck detector module."""

from types import SimpleNamespace

from backend.detectors.ai_detector import AIImageDetector, CapCheckDetector


RESULT_KEYS = {
    "module", "model", "status", "label", "human_score", "ai_score", "confidence", "device", "error"
}


def test_capcheck_detector_implements_interface() -> None:
    detector = CapCheckDetector()
    assert isinstance(detector, AIImageDetector) and detector._model is None


def test_missing_image_returns_standard_error_without_loading_model() -> None:
    detector = CapCheckDetector()
    result = detector.detect("does-not-exist.png")

    assert set(result) == RESULT_KEYS
    assert result["status"] == "error" and result["error"]
    assert result["module"] == "ai_detector"
    assert result["model"] == CapCheckDetector.MODEL_NAME
    assert all(result[key] is None for key in ("label", "human_score", "ai_score", "confidence"))
    assert detector._model is None and detector._processor is None


def test_unsupported_extension_returns_error_without_loading_model(tmp_path) -> None:
    image_path = tmp_path / "sample.gif"
    image_path.write_bytes(b"not examined because GIF is unsupported")
    detector = CapCheckDetector()

    result = detector.detect(image_path)

    assert result["status"] == "error" and "Supported image types" in result["error"]
    assert detector._model is None and detector._processor is None


def test_class_scores_follow_configured_label_ids() -> None:
    detector = CapCheckDetector()
    detector._model = SimpleNamespace(
        config=SimpleNamespace(id2label={0: "AI-generated", 1: "human"})
    )

    human_score, ai_score = detector._class_scores([0.2, 0.8])

    assert human_score == 0.8
    assert ai_score == 0.2
