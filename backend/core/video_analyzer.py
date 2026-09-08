"""Lightweight video analysis using sampled CapCheck frame scores."""

from __future__ import annotations

import json
import joblib
import numpy as np
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from backend.detectors.ai_detector.capcheck_detector import CapCheckDetector


SUPPORTED_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".webm"})
_detector: CapCheckDetector | None = None
_video_model = None
_VIDEO_MODEL_PATH = Path("models/video-baseline/best.joblib")


def is_supported_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def sample_times(duration: float, frames: int = 12) -> list[float]:
    if frames <= 0:
        raise ValueError("frames must be positive.")
    return [duration * index / (frames + 1) for index in range(1, frames + 1)] if duration > 0 else [0.0]


def temporal_features(scores: list[float]) -> list[float]:
    if not scores:
        raise ValueError("At least one frame score is required.")
    mean = statistics.fmean(scores)
    median = statistics.median(scores)
    low, high = min(scores), max(scores)
    diffs = [abs(b - a) for a, b in zip(scores, scores[1:])]
    mid = max(1, len(scores) // 2)
    q = statistics.quantiles(scores, n=100, method="inclusive") if len(scores) > 1 else [scores[0]] * 99
    return [mean, median, statistics.pstdev(scores), low, high, high - low,
            scores[-1] - scores[0], q[9], q[24], q[74], q[89],
            statistics.fmean(diffs) if diffs else 0.0, max(diffs, default=0.0),
            statistics.fmean(scores[:mid]), statistics.fmean(scores[mid:])]


def probe_video(path: str | Path) -> dict[str, Any]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration,format_name:stream=width,height,avg_frame_rate,nb_frames,codec_name", "-of", "json", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(completed.stdout)
    stream = data.get("streams", [{}])[0]
    rate = stream.get("avg_frame_rate", "0/0")
    numerator, denominator = (float(value) for value in rate.split("/", 1)) if "/" in rate else (float(rate), 1.0)
    return {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "width": stream.get("width"), "height": stream.get("height"),
        "fps": numerator / denominator if denominator else None,
        "frame_count": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "codec": stream.get("codec_name"), "container": data.get("format", {}).get("format_name"),
    }


def analyze_video(video_path: str | Path, frames: int = 12, detector: CapCheckDetector | None = None) -> dict[str, Any]:
    path = Path(video_path)
    base = {"pipeline": "video_forensic_analysis", "metadata": None, "sampled_frames": 0, "successful_frames": 0,
            "failed_frames": 0, "ai_score": None, "human_score": None, "label": None, "confidence": None,
            "model": None, "device": None, "frame_statistics": None, "errors": []}
    if not path.is_file() or not is_supported_video(path):
        base.update(status="failure", errors=[{"module": "input_validation", "error": "Unsupported or missing video file."}])
        return base
    try:
        metadata = probe_video(path)
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
        base.update(status="failure", errors=[{"module": "ffprobe", "error": str(exc)}])
        return base
    times = sample_times(metadata["duration"], frames)
    base.update(metadata=metadata, sampled_frames=len(times))
    active_detector = detector or _get_detector()
    scores: list[float] = []
    with tempfile.TemporaryDirectory(prefix="forensics-video-") as directory:
        for index, timestamp in enumerate(times):
            frame_path = Path(directory) / f"frame-{index}.jpg"
            try:
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(timestamp), "-i", str(path), "-frames:v", "1", str(frame_path)], capture_output=True, text=True, check=True)
                result = active_detector.detect(frame_path)
                if result.get("status") != "success" or result.get("ai_score") is None:
                    raise ValueError(str(result.get("error") or "Frame detector failed."))
                scores.append(float(result["ai_score"]))
                base["model"], base["device"] = result.get("model"), result.get("device")
            except (OSError, subprocess.CalledProcessError, ValueError) as exc:
                base["errors"].append({"frame": index, "error": str(exc)})
    base["successful_frames"], base["failed_frames"] = len(scores), len(times) - len(scores)
    if not scores:
        base["status"] = "failure"
        return base
    features = temporal_features(scores)
    model_state = joblib.load(_VIDEO_MODEL_PATH)
    video_model = model_state["model"]
    ai_score = float(video_model.predict_proba(np.asarray([features]))[0, 1])
    base.update(
        ai_score=ai_score,
        human_score=1 - ai_score,
        frame_statistics=dict(zip(
            ("mean", "median", "std", "min", "max", "range", "early_late_difference",
             "p10", "p25", "p75", "p90", "mean_abs_diff", "max_abs_diff",
             "first_half_mean", "second_half_mean"),
            features
        )),
        label="likely_ai" if ai_score >= .70 else "likely_human" if ai_score <= .30 else "inconclusive",
        confidence="high" if ai_score >= .85 or ai_score <= .15 else "medium" if ai_score >= .70 or ai_score <= .30 else "low",
        status="partial" if base["failed_frames"] else "success"
    )
    return base


def _get_detector() -> CapCheckDetector:
    global _detector
    if _detector is None:
        _detector = CapCheckDetector()
    return _detector
