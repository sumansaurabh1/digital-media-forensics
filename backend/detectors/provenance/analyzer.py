"""Local C2PA provenance inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.core.schemas import ForensicResult


MODULE_NAME = "provenance"


def analyze(image_path: str | Path) -> ForensicResult:
    """Inspect embedded C2PA provenance without making an authenticity judgment."""
    path = Path(image_path)

    try:
        if not path.is_file():
            raise FileNotFoundError(f"Image path does not exist: {path}")

        import c2pa

        try:
            reader = c2pa.Reader(str(path))
        except Exception as exc:
            if "ManifestNotFound" in str(exc):
                return ForensicResult(
                    module=MODULE_NAME,
                    status="success",
                    label="provenance_absent",
                    evidence=[{"type": "c2pa", "state": "absent"}],
                )
            raise

        try:
            embedded = reader.is_embedded()
        except Exception as exc:
            if "ManifestNotFound" in str(exc):
                embedded = False
            else:
                raise

        if not embedded:
            return ForensicResult(
                module=MODULE_NAME,
                status="success",
                label="provenance_absent",
                evidence=[{"type": "c2pa", "state": "absent"}],
            )

        manifest_store = json.loads(reader.json())
        evidence = _extract_evidence(manifest_store)

        return ForensicResult(
            module=MODULE_NAME,
            status="success",
            label="provenance_present",
            evidence=evidence,
        )

    except FileNotFoundError as exc:
        return ForensicResult(
            module=MODULE_NAME,
            status="error",
            error=f"Provenance analysis failed: {exc}",
        )
    except ImportError:
        return ForensicResult(
            module=MODULE_NAME,
            status="error",
            error="C2PA library is unavailable.",
        )
    except Exception as exc:
        return ForensicResult(
            module=MODULE_NAME,
            status="error",
            error=f"Provenance analysis failed: {exc}",
        )


def _extract_evidence(manifest_store: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract useful C2PA fields while keeping the result JSON serializable."""
    evidence: list[dict[str, Any]] = [
        {"type": "c2pa", "state": "present"}
    ]

    for key in (
        "active_manifest",
        "manifests",
        "validation_status",
        "validation_results",
    ):
        if key in manifest_store:
            evidence.append({
                "type": "c2pa_manifest_data",
                "name": key,
                "value": manifest_store[key],
            })

    return evidence