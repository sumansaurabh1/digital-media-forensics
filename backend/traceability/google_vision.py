"""Google Cloud Vision Web Detection adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


MODULE_NAME = "google_cloud_vision_web_detection"


def detect_web(image_path: str | Path) -> dict[str, Any]:
    """Submit local image bytes to Google Cloud Vision Web Detection."""
    path = Path(image_path)
    try:
        content = path.read_bytes()
    except (OSError, ValueError):
        return _error("Unable to read the image path.")

    vision = _vision_module()
    if vision is None:
        return _unavailable()

    try:
        client = vision.ImageAnnotatorClient()
        response = client.web_detection(image=vision.Image(content=content))
    except Exception:
        return _unavailable()

    if getattr(getattr(response, "error", None), "message", ""):
        return _error("Google Cloud Vision Web Detection request failed.")

    detection = getattr(response, "web_detection", None)
    return {
        "module": MODULE_NAME,
        "status": "success",
        "web_entities": [_entity(entity) for entity in getattr(detection, "web_entities", ()) or ()],
        "pages_with_matching_images": [
            _page(page) for page in getattr(detection, "pages_with_matching_images", ()) or ()
        ],
        "full_matching_image_urls": _urls(getattr(detection, "full_matching_images", ()) or ()),
        "partial_matching_image_urls": _urls(getattr(detection, "partial_matching_images", ()) or ()),
        "visually_similar_image_urls": _urls(getattr(detection, "visually_similar_images", ()) or ()),
    }


def _vision_module() -> Any | None:
    try:
        from google.cloud import vision
    except Exception:
        return None
    return vision


def _entity(entity: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "entity_id": getattr(entity, "entity_id", "") or None,
            "description": getattr(entity, "description", "") or None,
            "score": getattr(entity, "score", 0) or None,
        }.items()
        if value is not None
    }


def _page(page: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in {
            "url": getattr(page, "url", "") or None,
            "title": getattr(page, "page_title", "") or None,
        }.items()
        if value is not None
    }


def _urls(images: Any) -> list[str]:
    return [image.url for image in images if getattr(image, "url", "")]


def _unavailable() -> dict[str, Any]:
    return {
        "module": MODULE_NAME,
        "status": "unavailable",
        "error": "Google Cloud Vision Web Detection is unavailable.",
    }


def _error(message: str) -> dict[str, Any]:
    return {"module": MODULE_NAME, "status": "error", "error": message}
