"""Tests for image manipulation / tampering analysis.

All tests use synthetic temporary images — no internet, CUDA, or models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend.core.schemas import ForensicResult
from backend.detectors.manipulation.analyzer import analyze


# -- helpers -----------------------------------------------------------------

def _uniform(path: Path, fmt: str = "PNG", size=(128, 128)) -> Path:
    Image.new("RGB", size, (120, 130, 140)).save(path, format=fmt)
    return path


def _manipulated(path: Path) -> Path:
    """Dark background with a bright noisy patch → measurable signals."""
    base = np.full((256, 256, 3), 60, dtype=np.uint8)
    base[40:120, 40:120] = np.random.RandomState(42).randint(
        180, 255, (80, 80, 3), dtype=np.uint8)
    Image.fromarray(base).save(path, format="JPEG", quality=95)
    return path


# -- core tests --------------------------------------------------------------

def test_valid_synthetic_image_returns_success(tmp_path: Path) -> None:
    result = analyze(_uniform(tmp_path / "s.png"))
    assert isinstance(result, ForensicResult)
    assert result.module == "manipulation"
    assert result.status == "success"
    assert result.error is None


def test_png_input(tmp_path: Path) -> None:
    r = analyze(_uniform(tmp_path / "t.png", "PNG"))
    assert r.status == "success"
    assert 0.0 <= r.score <= 1.0


def test_jpeg_input(tmp_path: Path) -> None:
    p = tmp_path / "t.jpg"
    Image.new("RGB", (128, 128), (100, 100, 100)).save(p, format="JPEG", quality=85)
    r = analyze(p)
    assert r.status == "success"
    assert 0.0 <= r.score <= 1.0


def test_manipulated_image_output(tmp_path: Path) -> None:
    r = analyze(_manipulated(tmp_path / "m.jpg"))
    assert r.status == "success"
    assert 0.0 <= r.score <= 1.0
    assert r.confidence is not None
    names = {e["name"] for e in r.evidence if e.get("type") == "signal"}
    assert names == {"error_level_analysis", "noise_residual", "edge_texture"}
    for e in r.evidence:
        if e.get("type") == "signal":
            assert 0.0 <= e["score"] <= 1.0


def test_suspicious_regions_valid_coordinates(tmp_path: Path) -> None:
    r = analyze(_manipulated(tmp_path / "m.jpg"))
    for e in r.evidence:
        if e.get("type") != "suspicious_regions":
            continue
        assert isinstance(e["count"], int) and e["count"] >= 0
        for reg in e["regions"]:
            assert reg["x"] >= 0 and reg["y"] >= 0
            assert reg["width"] > 0 and reg["height"] > 0
            assert reg["x"] + reg["width"] <= 256
            assert reg["y"] + reg["height"] <= 256


def test_missing_file_error(tmp_path: Path) -> None:
    r = analyze(tmp_path / "nope.png")
    assert r.status == "error" and r.error and r.module == "manipulation"


def test_invalid_file_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.png"
    p.write_text("not an image")
    r = analyze(p)
    assert r.status == "error" and r.error and r.module == "manipulation"


def test_json_serializable(tmp_path: Path) -> None:
    r = analyze(_manipulated(tmp_path / "m.jpg"))
    payload = json.loads(r.model_dump_json())
    assert payload["module"] == "manipulation"
    assert isinstance(payload["score"], float)
    assert isinstance(payload["evidence"], list)
    json.dumps(payload)  # round-trip


def test_grayscale_input(tmp_path: Path) -> None:
    p = tmp_path / "g.png"
    Image.new("L", (128, 128), 128).save(p)
    r = analyze(p)
    assert r.status == "success" and 0.0 <= r.score <= 1.0


def test_rgba_input(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    Image.new("RGBA", (128, 128), (100, 150, 200, 128)).save(p)
    r = analyze(p)
    assert r.status == "success" and 0.0 <= r.score <= 1.0


def test_very_small_image(tmp_path: Path) -> None:
    p = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), "red").save(p)
    r = analyze(p)
    assert r.status == "success" and r.score == 0.0 and r.label == "image_too_small"


def test_forensic_result_schema(tmp_path: Path) -> None:
    r = analyze(_uniform(tmp_path / "s.png"))
    assert isinstance(r, ForensicResult)
    for attr in ("status", "score", "label", "confidence", "evidence", "artifacts", "error"):
        assert hasattr(r, attr)
