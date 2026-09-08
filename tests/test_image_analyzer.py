"""Focused tests for the image-forensic orchestration layer."""

import json

from PIL import Image

from backend.core import image_analyzer


def _image_path(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), "blue").save(path)
    return path


def _stubbed_stages(monkeypatch):
    calls = []
    def stage(name):
        def run(path):
            calls.append(name)
            return {"module": name, "status": "success"}
        return run
    monkeypatch.setattr(image_analyzer, "metadata_analyze", stage("metadata"))
    monkeypatch.setattr(image_analyzer, "calculate_sha256", stage("sha256"))
    monkeypatch.setattr(image_analyzer, "calculate_phash", stage("perceptual_hash"))
    monkeypatch.setattr(image_analyzer, "detect_web", stage("traceability"))
    monkeypatch.setattr(image_analyzer, "manipulation_analyze", stage("manipulation"))
    class Detector:
        def detect(self, path):
            calls.append("ai_detection")
            return {"module": "ai_detector", "status": "success"}
    monkeypatch.setattr(image_analyzer, "CapCheckDetector", Detector)
    return calls


def test_successful_orchestration_and_json_serialization(tmp_path, monkeypatch) -> None:
    calls = _stubbed_stages(monkeypatch)

    result = image_analyzer.analyze_image(_image_path(tmp_path))

    assert result["status"] == "success" and result["errors"] == []
    assert set(result) == {"pipeline", "status", "metadata", "fingerprints", "traceability", "ai_detection", "manipulation", "provenance", 	"errors"}
    assert calls == ["metadata", "sha256", "perceptual_hash", "traceability", "ai_detection", "manipulation"]
    json.dumps(result)


def test_stage_failure_is_isolated(tmp_path, monkeypatch) -> None:
    def fail(path):
        raise RuntimeError("metadata unavailable")
    monkeypatch.setattr(image_analyzer, "metadata_analyze", fail)
    monkeypatch.setattr(
        image_analyzer, "detect_web",
        lambda path: {"module": "google_cloud_vision_web_detection", "status": "unavailable"},
    )
    class Detector:
        def detect(self, path):
            return {"module": "ai_detector", "status": "success"}
    monkeypatch.setattr(image_analyzer, "CapCheckDetector", Detector)

    result = image_analyzer.analyze_image(_image_path(tmp_path))

    assert result["status"] == "partial"
    assert result["metadata"]["status"] == "error"
    assert result["fingerprints"]["sha256"]["status"] == "success"
    assert result["manipulation"]["status"] == "success"
    assert result["errors"] == [{"module": "metadata", "error": "metadata unavailable"}]


def test_missing_input_fails_before_stages_run(tmp_path, monkeypatch) -> None:
    called = False
    def fail_if_called(path):
        nonlocal called
        called = True
        raise AssertionError("stage should not run")
    monkeypatch.setattr(image_analyzer, "metadata_analyze", fail_if_called)

    result = image_analyzer.analyze_image(tmp_path / "missing.png")

    assert result["status"] == "failure" and result["metadata"] is None
    assert result["errors"][0]["module"] == "input_validation"
    assert not called
