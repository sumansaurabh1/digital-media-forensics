"""Tests for exact and perceptual image fingerprints."""

import hashlib

from PIL import Image, ImageEnhance

from backend.fingerprint.perceptual import calculate_phash, hamming_distance
from backend.fingerprint.sha256 import calculate_sha256


def _image_path(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), "blue").save(path)
    return path


def test_sha256_is_exact_and_deterministic(tmp_path) -> None:
    path = tmp_path / "known.bin"
    path.write_bytes(b"forensic test data")

    first = calculate_sha256(path)
    second = calculate_sha256(path)

    assert first.artifacts[0]["value"] == "0ad94aa233c33d5b0d9a4cf23889c41bb3679505d9e4f06a0e4bfe9b50c86d8d"
    assert first.artifacts[0]["value"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert first.artifacts == second.artifacts


def test_sha256_changes_when_file_bytes_change(tmp_path) -> None:
    path = _image_path(tmp_path)
    original = calculate_sha256(path).artifacts[0]["value"]
    path.write_bytes(path.read_bytes() + b"changed")

    assert calculate_sha256(path).artifacts[0]["value"] != original


def test_sha256_missing_file_returns_standard_error(tmp_path) -> None:
    result = calculate_sha256(tmp_path / "missing.png")

    assert result.status == "error" and result.error


def test_phash_is_deterministic_and_identical_images_have_no_distance(tmp_path) -> None:
    path = _image_path(tmp_path)

    first = calculate_phash(path)
    second = calculate_phash(path)
    first_hash = first.artifacts[0]["value"]

    assert first.status == "success" and first_hash == second.artifacts[0]["value"]
    assert hamming_distance(first_hash, first_hash) == 0


def test_phash_of_visually_similar_images_has_small_distance(tmp_path) -> None:
    original = tmp_path / "original.png"
    similar = tmp_path / "similar.png"
    image = Image.new("L", (32, 32), "white")
    image.paste("black", (8, 8, 24, 24))
    image.save(original)
    ImageEnhance.Brightness(image).enhance(0.95).save(similar)

    original_hash = calculate_phash(original).artifacts[0]["value"]
    similar_hash = calculate_phash(similar).artifacts[0]["value"]

    assert hamming_distance(original_hash, similar_hash) <= 8


def test_phash_invalid_or_missing_image_returns_standard_error(tmp_path) -> None:
    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image")

    assert calculate_phash(tmp_path / "missing.png").status == "error"
    assert calculate_phash(invalid).status == "error"
