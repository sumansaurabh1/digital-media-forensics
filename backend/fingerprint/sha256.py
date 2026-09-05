"""Exact byte-level SHA-256 fingerprinting for forensic files."""

from __future__ import annotations

import hashlib
from pathlib import Path

from backend.core.schemas import ForensicResult


MODULE_NAME = "sha256"
_CHUNK_SIZE = 1024 * 1024


def calculate_sha256(image_path: str | Path) -> ForensicResult:
    """Return the SHA-256 digest of exact file bytes using streamed reads."""
    path = Path(image_path)
    try:
        if not path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {path}")

        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        return ForensicResult(
            module=MODULE_NAME,
            status="success",
            label="exact_file_fingerprint",
            evidence=[{
                "type": "fingerprint",
                "algorithm": "SHA-256",
                "value": value,
                "meaning": "Identifies these exact file bytes, not visually similar images.",
            }],
            artifacts=[{"name": "sha256", "value": value}],
        )
    except (FileNotFoundError, OSError) as exc:
        return ForensicResult(module=MODULE_NAME, status="error", error=f"SHA-256 calculation failed: {exc}")
