"""Safe extraction of image properties and available EXIF metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from backend.core.schemas import ForensicResult


MODULE_NAME = "metadata"
_EXIF_FIELDS = {
    271: "camera_make",
    272: "camera_model",
    274: "orientation",
    305: "software",
    306: "datetime",
    36867: "datetime_original",
    36868: "datetime_digitized",
}
_GPS_INFO_TAG = 34853


def analyze(image_path: str | Path) -> ForensicResult:
    """Return image properties and available metadata without judging authenticity."""
    path = Path(image_path)
    try:
        if not path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {path}")

        with Image.open(path) as image:
            properties = {
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "file_size": path.stat().st_size,
            }
            try:
                exif = image.getexif()
                metadata = {
                    name: _json_value(exif[tag])
                    for tag, name in _EXIF_FIELDS.items()
                    if tag in exif
                }
                if _GPS_INFO_TAG in exif:
                    metadata["gps_present"] = True
                state = "present" if exif else "absent"
            except Exception as exc:
                return ForensicResult(
                    module=MODULE_NAME,
                    status="success",
                    label="metadata_parsing_failed",
                    evidence=[
                        {"type": "image_properties", "values": properties},
                        {"type": "metadata", "state": "parsing_failed", "error": str(exc)},
                    ],
                )

        return ForensicResult(
            module=MODULE_NAME,
            status="success",
            label=f"metadata_{state}",
            evidence=[
                {"type": "image_properties", "values": properties},
                {"type": "metadata", "state": state, "values": metadata},
            ],
        )
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return ForensicResult(module=MODULE_NAME, status="error", error=f"Metadata analysis failed: {exc}")
    except Exception as exc:
        return ForensicResult(module=MODULE_NAME, status="error", error=f"Metadata analysis failed: {exc}")


def _json_value(value: Any) -> Any:
    """Normalize Pillow EXIF values into JSON-compatible primitives."""
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
