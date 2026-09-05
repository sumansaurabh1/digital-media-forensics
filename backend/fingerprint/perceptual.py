"""Perceptual hashing for later visual-similarity comparisons."""

from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

from backend.core.schemas import ForensicResult


MODULE_NAME = "perceptual_hash"


def calculate_phash(image_path: str | Path) -> ForensicResult:
    """Return an imagehash pHash; it aids similarity, not provenance proof."""
    path = Path(image_path)
    try:
        if not path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {path}")
        with Image.open(path) as image:
            value = str(imagehash.phash(image))
        return ForensicResult(
            module=MODULE_NAME,
            status="success",
            label="perceptual_fingerprint",
            evidence=[{
                "type": "fingerprint",
                "algorithm": "imagehash.phash",
                "value": value,
                "meaning": "A visual-similarity aid, not proof of a shared original source.",
            }],
            artifacts=[{"name": "phash", "value": value, "algorithm": "imagehash.phash"}],
        )
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return ForensicResult(module=MODULE_NAME, status="error", error=f"Perceptual hash calculation failed: {exc}")
    except Exception as exc:
        return ForensicResult(module=MODULE_NAME, status="error", error=f"Perceptual hash calculation failed: {exc}")


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Return differing bits between compatible hexadecimal perceptual hashes."""
    if not isinstance(hash_a, str) or not isinstance(hash_b, str):
        raise ValueError("Perceptual hashes must be hexadecimal strings.")
    if len(hash_a) != len(hash_b):
        raise ValueError("Perceptual hashes must have the same length.")
    try:
        int(hash_a, 16)
        int(hash_b, 16)
        return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
    except (TypeError, ValueError) as exc:
        raise ValueError("Perceptual hashes must be hexadecimal strings.") from exc
