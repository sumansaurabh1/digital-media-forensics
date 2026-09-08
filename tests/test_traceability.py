"""Tests for the Google Cloud Vision Web Detection adapter."""

import json
from types import SimpleNamespace

from backend.traceability import google_vision
from backend.core import image_analyzer
from PIL import Image


def _vision(client):
    class Image:
        def __init__(self, content):
            self.content = content

    return SimpleNamespace(Image=Image, ImageAnnotatorClient=lambda: client)


def _response(**values):
    return SimpleNamespace(
        error=SimpleNamespace(message=""),
        web_detection=SimpleNamespace(**values),
    )


def test_detect_web_normalizes_all_returned_match_types(tmp_path, monkeypatch) -> None:
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"image bytes")
    response = _response(
        web_entities=[SimpleNamespace(entity_id="/m/cat", description="Cat", score=0.91)],
        pages_with_matching_images=[
            SimpleNamespace(url="https://example.test/page", page_title="Example page"),
            SimpleNamespace(url="https://example.test/untitled", page_title=""),
        ],
        full_matching_images=[SimpleNamespace(url="https://example.test/full.jpg")],
        partial_matching_images=[SimpleNamespace(url="https://example.test/partial.jpg")],
        visually_similar_images=[SimpleNamespace(url="https://example.test/similar.jpg")],
    )

    class Client:
        def web_detection(self, image):
            assert image.content == b"image bytes"
            return response

    monkeypatch.setattr(google_vision, "_vision_module", lambda: _vision(Client()))

    assert google_vision.detect_web(image) == {
        "module": "google_cloud_vision_web_detection",
        "status": "success",
        "web_entities": [{"entity_id": "/m/cat", "description": "Cat", "score": 0.91}],
        "pages_with_matching_images": [
            {"url": "https://example.test/page", "title": "Example page"},
            {"url": "https://example.test/untitled"},
        ],
        "full_matching_image_urls": ["https://example.test/full.jpg"],
        "partial_matching_image_urls": ["https://example.test/partial.jpg"],
        "visually_similar_image_urls": ["https://example.test/similar.jpg"],
    }


def test_detect_web_returns_empty_json_serializable_result_for_no_matches(tmp_path, monkeypatch) -> None:
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"image bytes")

    class Client:
        def web_detection(self, image):
            return _response()

    monkeypatch.setattr(google_vision, "_vision_module", lambda: _vision(Client()))

    result = google_vision.detect_web(str(image))

    assert result["status"] == "success"
    assert all(value == [] for key, value in result.items() if key not in {"module", "status"})
    json.dumps(result)


def test_detect_web_returns_safe_unavailable_result_for_client_or_api_failure(tmp_path, monkeypatch) -> None:
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"image bytes")

    class Client:
        def web_detection(self, image):
            raise RuntimeError("token=super-secret credential-path=C:/private.json")

    monkeypatch.setattr(google_vision, "_vision_module", lambda: _vision(Client()))
    result = google_vision.detect_web(image)

    assert result == {
        "module": "google_cloud_vision_web_detection",
        "status": "unavailable",
        "error": "Google Cloud Vision Web Detection is unavailable.",
    }
    assert "super-secret" not in json.dumps(result)


def test_detect_web_returns_safe_unavailable_result_when_client_cannot_initialize(tmp_path, monkeypatch) -> None:
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"image bytes")

    class Vision:
        class Image:
            def __init__(self, content):
                self.content = content

        @staticmethod
        def ImageAnnotatorClient():
            raise RuntimeError("secret token")

    monkeypatch.setattr(google_vision, "_vision_module", lambda: Vision)

    assert google_vision.detect_web(image)["status"] == "unavailable"


def test_detect_web_returns_error_for_missing_or_invalid_path(tmp_path) -> None:
    assert google_vision.detect_web(tmp_path / "missing.jpg") == {
        "module": "google_cloud_vision_web_detection",
        "status": "error",
        "error": "Unable to read the image path.",
    }


def _image_path(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), "blue").save(path)
    return path


def _pipeline_stages(monkeypatch, traceability):
    calls = []
    def stage(name):
        def run(path):
            calls.append(name)
            return {"module": name, "status": "success"}
        return run
    monkeypatch.setattr(image_analyzer, "metadata_analyze", stage("metadata"))
    monkeypatch.setattr(image_analyzer, "calculate_sha256", stage("sha256"))
    monkeypatch.setattr(image_analyzer, "calculate_phash", stage("perceptual_hash"))
    monkeypatch.setattr(image_analyzer, "detect_web", traceability)
    monkeypatch.setattr(image_analyzer, "manipulation_analyze", stage("manipulation"))

    class Detector:
        def detect(self, path):
            calls.append("ai_detection")
            return {"module": "ai_detector", "status": "success"}

    monkeypatch.setattr(image_analyzer, "CapCheckDetector", Detector)
    monkeypatch.setattr(image_analyzer, "_ai_detector", None)
    return calls


def test_successful_traceability_is_preserved_at_the_pipeline_top_level(tmp_path, monkeypatch) -> None:
    traceability = {"module": "google_cloud_vision_web_detection", "status": "success", "pages_with_matching_images": [{"url": "https://example.test"}]}
    calls = _pipeline_stages(monkeypatch, lambda path: traceability)

    result = image_analyzer.analyze_image(_image_path(tmp_path))

    assert result["traceability"] is traceability
    assert result["status"] == "success" and result["errors"] == []
    assert calls == ["metadata", "sha256", "perceptual_hash", "ai_detection", "manipulation"]


def test_unavailable_traceability_remains_optional(tmp_path, monkeypatch) -> None:
    traceability = {"module": "google_cloud_vision_web_detection", "status": "unavailable", "error": "unavailable"}
    _pipeline_stages(monkeypatch, lambda path: traceability)

    result = image_analyzer.analyze_image(_image_path(tmp_path))

    assert result["traceability"] is traceability
    assert result["status"] == "success" and result["errors"] == []


def test_traceability_errors_and_exceptions_are_isolated(tmp_path, monkeypatch) -> None:
    error = {"module": "google_cloud_vision_web_detection", "status": "error", "error": "request failed"}
    calls = _pipeline_stages(monkeypatch, lambda path: error)

    result = image_analyzer.analyze_image(_image_path(tmp_path))

    assert result["traceability"] is error
    assert result["status"] == "partial"
    assert result["errors"] == [{"module": "traceability", "error": "request failed"}]
    assert calls[-2:] == ["ai_detection", "manipulation"]

    def fail(path):
        raise RuntimeError("token=secret")

    calls = _pipeline_stages(monkeypatch, fail)
    result = image_analyzer.analyze_image(_image_path(tmp_path))

    assert result["traceability"]["error"] == "Traceability request failed."
    assert "secret" not in json.dumps(result)
    assert calls[-2:] == ["ai_detection", "manipulation"]
