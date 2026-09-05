"""Tests for the CapCheck image-analysis HTTP integration."""

from backend.api import routes
from backend.main import app
from fastapi.testclient import TestClient


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82"
)


class StubDetector:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.image_path = None

    def detect(self, image_path):
        self.image_path = image_path
        return self.result


def test_analyze_image_returns_capcheck_result_and_removes_upload(monkeypatch) -> None:
    expected = {
        "module": "ai_detector", "model": "capcheck/ai-human-generated-image-detection",
        "status": "success", "label": "ai", "human_score": 0.1, "ai_score": 0.9,
        "confidence": "high", "device": "cpu", "error": None,
    }
    stub = StubDetector(expected)
    monkeypatch.setattr(routes, "detector", stub)

    response = TestClient(app).post(
        "/analyze/image", files={"image": ("sample.png", PNG_BYTES, "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == expected
    assert stub.image_path is not None and not stub.image_path.exists()


def test_analyze_image_rejects_unsupported_upload(monkeypatch) -> None:
    stub = StubDetector({})
    monkeypatch.setattr(routes, "detector", stub)

    response = TestClient(app).post(
        "/analyze/image", files={"image": ("sample.gif", b"GIF89a", "image/gif")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Supported image types: PNG, JPG, JPEG, WEBP."
    assert stub.image_path is None


def test_analyze_image_returns_detector_error_result(monkeypatch) -> None:
    expected = {
        "module": "ai_detector", "model": "capcheck/ai-human-generated-image-detection",
        "status": "error", "label": None, "human_score": None, "ai_score": None,
        "confidence": None, "device": "cpu", "error": "Model is not available.",
    }
    monkeypatch.setattr(routes, "detector", StubDetector(expected))

    response = TestClient(app).post(
        "/analyze/image", files={"image": ("sample.png", PNG_BYTES, "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_analyze_image_handles_unexpected_detector_failure(monkeypatch) -> None:
    class FailingDetector:
        def detect(self, image_path):
            raise RuntimeError("unavailable")

    monkeypatch.setattr(routes, "detector", FailingDetector())

    response = TestClient(app).post(
        "/analyze/image", files={"image": ("sample.png", PNG_BYTES, "image/png")}
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Image analysis failed."}
