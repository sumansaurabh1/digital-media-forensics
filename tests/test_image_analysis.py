"""Tests for the image-forensic analysis HTTP integration."""

from backend.api import routes
from backend.main import app
from fastapi.testclient import TestClient
import json


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


class StubAnalyzer:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.image_path = None

    def __call__(self, image_path):
        self.image_path = image_path
        return self.result


def test_health_and_frontend_remain_available() -> None:
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "healthy"}
    assert "Digital Media Forensics" in client.get("/").text


def test_analyze_image_returns_orchestrator_result_and_removes_upload(monkeypatch) -> None:
    expected = {
        "pipeline": "image_forensic_analysis", "status": "success",
        "metadata": {"module": "metadata", "status": "success"},
        "fingerprints": {"sha256": {}, "perceptual_hash": {}},
        "ai_detection": {"module": "ai_detector", "status": "success"},
        "manipulation": {"module": "manipulation", "status": "success"}, "errors": [],
    }
    stub = StubAnalyzer(expected.copy())
    monkeypatch.setattr(routes, "analyze_forensic_image", stub)

    response = TestClient(app).post(
        "/analyze/image", files={"image": ("sample.png", PNG_BYTES, "image/png")}
    )

    assert response.status_code == 200
    payload = response.json()
    assert {key: payload[key] for key in expected} == expected
    assert "evidence_report" in payload
    json.dumps(payload["evidence_report"])
    assert stub.image_path is not None and not stub.image_path.exists()


def test_analyze_image_rejects_unsupported_upload(monkeypatch) -> None:
    stub = StubAnalyzer({})
    monkeypatch.setattr(routes, "analyze_forensic_image", stub)

    response = TestClient(app).post(
        "/analyze/image", files={"image": ("sample.gif", b"GIF89a", "image/gif")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Supported image types: PNG, JPG, JPEG, WEBP."
    assert stub.image_path is None


def test_analyze_image_returns_orchestrator_error_result(monkeypatch) -> None:
    expected = {
        "pipeline": "image_forensic_analysis", "status": "partial",
        "metadata": {}, "fingerprints": {"sha256": {}, "perceptual_hash": {}},
        "ai_detection": {"module": "ai_detector", "status": "error", "error": "Model is not available."},
        "manipulation": {}, "errors": [{"module": "ai_detection", "error": "Model is not available."}],
    }
    monkeypatch.setattr(routes, "analyze_forensic_image", StubAnalyzer(expected.copy()))

    response = TestClient(app).post(
        "/analyze/image", files={"image": ("sample.png", PNG_BYTES, "image/png")}
    )

    assert response.status_code == 200
    payload = response.json()
    assert {key: payload[key] for key in expected} == expected
    assert "evidence_report" in payload
    json.dumps(payload["evidence_report"])


def test_analyze_image_rejects_invalid_image_content(monkeypatch) -> None:
    stub = StubAnalyzer({})
    monkeypatch.setattr(routes, "analyze_forensic_image", stub)

    response = TestClient(app).post(
        "/analyze/image", files={"image": ("sample.png", b"not an image", "image/png")}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Image analysis failed."}
    assert stub.image_path is None


def test_analyze_image_handles_unexpected_orchestration_failure(monkeypatch) -> None:
    def fail(image_path):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(routes, "analyze_forensic_image", fail)

    response = TestClient(app).post(
        "/analyze/image", files={"image": ("sample.png", PNG_BYTES, "image/png")}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Image analysis failed."}
