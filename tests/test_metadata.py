"""Tests for image metadata analysis."""

import json

from PIL import Image

from backend.detectors.metadata.analyzer import analyze


def test_metadata_extracts_basic_properties_without_exif(tmp_path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (3, 2), "red").save(path)

    result = analyze(path)
    properties = result.evidence[0]["values"]

    assert result.status == "success" and result.label == "metadata_absent"
    assert properties == {"format": "PNG", "width": 3, "height": 2, "mode": "RGB", "file_size": path.stat().st_size}
    assert json.loads(result.json())["evidence"][1]["state"] == "absent"


def test_metadata_extracts_available_exif_fields(tmp_path) -> None:
    path = tmp_path / "camera.jpg"
    exif = Image.Exif()
    exif[271] = "Example Camera Co."
    exif[272] = "Model A"
    exif[305] = "Test Editor"
    Image.new("RGB", (2, 2), "blue").save(path, exif=exif)

    result = analyze(path)
    metadata = result.evidence[1]["values"]

    assert result.status == "success" and result.label == "metadata_present"
    assert metadata["camera_make"] == "Example Camera Co."
    assert metadata["camera_model"] == "Model A"
    assert metadata["software"] == "Test Editor"


def test_metadata_missing_file_returns_standard_error(tmp_path) -> None:
    result = analyze(tmp_path / "missing.png")

    assert result.status == "error" and result.error


def test_metadata_invalid_image_returns_standard_error(tmp_path) -> None:
    path = tmp_path / "invalid.png"
    path.write_text("not an image")

    result = analyze(path)

    assert result.status == "error" and result.error
