"""Orchestrate existing image-forensic modules without interpreting their results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError

from backend.core.schemas import ForensicResult
from backend.detectors.ai_detector.capcheck_detector import CapCheckDetector
from backend.detectors.manipulation.analyzer import analyze as manipulation_analyze
from backend.detectors.metadata.analyzer import analyze as metadata_analyze
from backend.fingerprint.perceptual import calculate_phash
from backend.fingerprint.sha256 import calculate_sha256

_ai_detector: CapCheckDetector | None = None


def analyze_image(image_path: str | Path) -> dict[str, Any]:
    """Run each available image-forensic stage and return their uncombined output."""
    path = Path(image_path)
    invalid = _validation_error(path)
    if invalid:
        return {
            "pipeline": "image_forensic_analysis", "status": "failure",
            "metadata": None, "fingerprints": {"sha256": None, "perceptual_hash": None},
            "ai_detection": None, "manipulation": None,
            "errors": [{"module": "input_validation", "error": invalid}],
        }

    errors: list[dict[str, str]] = []
    def run(name: str, stage: Callable[[Path], Any]) -> Any:
        try:
            result = _json_result(stage(path))
            if result.get("status") == "error":
                errors.append({"module": name, "error": str(result.get("error") or "Stage failed")})
            return result
        except Exception as exc:
            errors.append({"module": name, "error": str(exc)})
            return _json_result(ForensicResult(module=name, status="error", error=str(exc)))

    metadata = run("metadata", metadata_analyze)
    sha256 = run("sha256", calculate_sha256)
    perceptual_hash = run("perceptual_hash", calculate_phash)
    ai_detection = run("ai_detection", lambda value: _get_ai_detector().detect(value))
    manipulation = run("manipulation", manipulation_analyze)
    return {
        "pipeline": "image_forensic_analysis",
        "status": "partial" if errors else "success",
        "metadata": metadata,
        "fingerprints": {"sha256": sha256, "perceptual_hash": perceptual_hash},
        "ai_detection": ai_detection,
        "manipulation": manipulation,
        "errors": errors,
    }


def _validation_error(path: Path) -> str | None:
    try:
        if not path.is_file():
            return f"Image path does not exist: {path}"
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        return f"Unable to read image: {exc}"
    return None


def _get_ai_detector() -> CapCheckDetector:
    global _ai_detector
    if _ai_detector is None:
        _ai_detector = CapCheckDetector()
    return _ai_detector


def _json_result(result: Any) -> dict[str, Any]:
    return result.model_dump(mode="json") if isinstance(result, ForensicResult) else result
