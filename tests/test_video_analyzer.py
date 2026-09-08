from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.core import video_analyzer
from backend.main import app


def test_supported_video_extensions() -> None:
    assert video_analyzer.is_supported_video("clip.MP4")
    assert not video_analyzer.is_supported_video("clip.mkv")


def test_frame_sampling_is_even() -> None:
    assert video_analyzer.sample_times(13, 3) == [3.25, 6.5, 9.75]


def test_video_result_structure(tmp_path, monkeypatch) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(video_analyzer, "probe_video", lambda _: {"duration": 2, "width": 10, "height": 20, "fps": 25, "frame_count": 50, "codec": "h264", "container": "mp4"})
    monkeypatch.setattr(video_analyzer.subprocess, "run", lambda *args, **kwargs: SimpleNamespace())
    detector = SimpleNamespace(detect=lambda _: {"status": "success", "ai_score": .8, "model": "capcheck", "device": "cpu"})

    result = video_analyzer.analyze_video(video, frames=2, detector=detector)

    assert {"pipeline", "status", "metadata", "sampled_frames", "successful_frames", "failed_frames", "ai_score", "human_score", "label", "confidence", "model", "device", "errors"} <= set(result)
    assert result["successful_frames"] == 2 and result["label"] == "likely_ai"


def test_video_endpoint_rejects_invalid_extension() -> None:
    response = TestClient(app).post("/analyze/video", files={"video": ("clip.mkv", b"x", "video/x-matroska")})

    assert response.status_code == 400
