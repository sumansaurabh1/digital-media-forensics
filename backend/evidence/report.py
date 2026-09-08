
"""Build a cautious evidence summary from forensic analysis results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SOURCE_TIMELINE_LIMITATION = "Source/timeline was not established by the available analysis."
_AI_LIMITATION = (
    "AI detection is one forensic signal and is not, by itself, a definitive "
    "authenticity determination."
)
_MANIPULATION_LIMITATION = (
    "Signal-based manipulation analysis does not by itself establish that an "
    "image was definitively edited."
)


def generate_evidence_report(result: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize available forensic output without altering or extending it."""
    source: Mapping[str, Any] = result if isinstance(result, Mapping) else {}
    status = source.get("status")
    ai_result = source.get("ai_detection")
    manipulation_result = source.get("manipulation")
    metadata_result = source.get("metadata")
    fingerprints = source.get("fingerprints")
    traceability = source.get("traceability")
    provenance = source.get("provenance")
    errors = source.get("errors") or []

    observed = _observations(
        status, ai_result, manipulation_result, metadata_result, fingerprints, provenance
    )
    interpretation = _interpretations(status, ai_result, manipulation_result, provenance)
    limitations = [_AI_LIMITATION, _MANIPULATION_LIMITATION, SOURCE_TIMELINE_LIMITATION]

    return {
        "investigation_summary": _summary(status),
        "overall_assessment": _assessment(status),
        "key_findings": _findings(
            status, ai_result, manipulation_result, metadata_result, fingerprints
        ),
        "supporting_evidence": {
            "observed": observed,
            "interpretation": interpretation,
            "limitations": limitations,
        },
        "warnings_limitations": limitations + [
            "SHA-256 identifies exact file bytes; a perceptual hash supports visual similarity comparisons and is not proof of a shared original source."
        ],
        "fingerprints": fingerprints,
        "traceability": traceability,
        "metadata_summary": metadata_result,
        "ai_detection": ai_result,
        "manipulation": manipulation_result,
        "provenance": provenance,
        "pipeline_errors": errors,
    }


def _summary(status: Any) -> str:
    if status == "success":
        return "Available image-forensic stages completed successfully."
    if status == "partial":
        return "The image-forensic pipeline completed partially; some analysis stages were unavailable or failed."
    if status == "failure":
        return "The image-forensic pipeline failed; available findings are incomplete."
    return "The available image-forensic result does not provide a complete pipeline status."


def _assessment(status: Any) -> str:
    if status in {"partial", "failure"}:
        return "The investigation is incomplete because one or more forensic stages failed."
    return (
        "The available forensic signals provide evidence for further examination but do not "
        "establish authenticity or manipulation with certainty."
    )


def _observations(
    status: Any,
    ai_result: Any,
    manipulation_result: Any,
    metadata_result: Any,
    fingerprints: Any,
    provenance: Any,
) -> list[str]:
    observations: list[str] = []

    if status in {"success", "partial", "failure"}:
        observations.append(f"Pipeline status: {status}.")

    if isinstance(ai_result, Mapping) and ai_result.get("status") == "success":
        fields = _present_fields(
            ai_result, ("label", "human_score", "ai_score", "confidence", "model")
        )
        if fields:
            observations.append(f"AI detector reported {fields}.")

    if isinstance(manipulation_result, Mapping) and manipulation_result.get("status") == "success":
        fields = _present_fields(
            manipulation_result, ("label", "score", "confidence", "evidence")
        )
        if fields:
            observations.append(f"Manipulation analysis reported {fields}.")

    if isinstance(metadata_result, Mapping):
        label = metadata_result.get("label")
        if label == "metadata_absent":
            observations.append("Metadata analysis reported that EXIF metadata was absent.")
        elif metadata_result.get("status") == "success":
            observations.append("Metadata analysis produced an available result.")

    if isinstance(fingerprints, Mapping):
        if fingerprints.get("sha256") is not None:
            observations.append("A SHA-256 fingerprint result is available.")
        if fingerprints.get("perceptual_hash") is not None:
            observations.append("A perceptual-hash result is available.")

    if isinstance(provenance, Mapping) and provenance.get("status") == "success":
        if provenance.get("label") == "provenance_present":
            observations.append("C2PA provenance data is embedded in the image.")
        elif provenance.get("label") == "provenance_absent":
            observations.append("C2PA provenance data was not present in the image.")

    return observations


def _interpretations(
    status: Any,
    ai_result: Any,
    manipulation_result: Any,
    provenance: Any,
) -> list[str]:
    interpretation: list[str] = []

    if status in {"partial", "failure"}:
        interpretation.append(
            "Some analysis stages were unavailable or failed, so the available evidence is incomplete."
        )

    if isinstance(ai_result, Mapping) and ai_result.get("status") == "success":
        if ai_result.get("label") == "ai":
            interpretation.append(
                "The AI-image detector produced a signal consistent with likely AI-generated content."
            )
        elif ai_result.get("label") == "human":
            interpretation.append(
                "The AI-image detector produced a signal consistent with likely human-origin content."
            )

    if isinstance(manipulation_result, Mapping) and manipulation_result.get("status") == "success":
        interpretation.append(
            "The manipulation analysis identified image-level inconsistency signals that may warrant further examination."
        )

    if isinstance(provenance, Mapping) and provenance.get("status") == "success":
        if provenance.get("label") == "provenance_present":
            interpretation.append(
                "Embedded C2PA provenance provides attribution evidence about the image's creation history; it does not by itself establish authenticity."
            )
        elif provenance.get("label") == "provenance_absent":
            interpretation.append(
                "No embedded C2PA provenance was found; absence of provenance does not establish that an image is fake or manipulated."
            )

    return interpretation


def _findings(
    status: Any,
    ai_result: Any,
    manipulation_result: Any,
    metadata_result: Any,
    fingerprints: Any,
) -> list[str]:
    findings: list[str] = []

    if status in {"partial", "failure"}:
        findings.append(f"Pipeline status is {status}; the investigation is incomplete.")
    elif status == "success":
        findings.append("Pipeline completed successfully.")

    if (
        isinstance(ai_result, Mapping)
        and ai_result.get("status") == "success"
        and ai_result.get("label") in {"ai", "human"}
    ):
        findings.append(f"AI detector label: {ai_result['label']}.")

    if isinstance(manipulation_result, Mapping) and manipulation_result.get("status") == "success":
        fields = _present_fields(manipulation_result, ("label", "score"))
        if fields:
            findings.append(f"Manipulation analysis result: {fields}.")

    if isinstance(metadata_result, Mapping) and metadata_result.get("label") == "metadata_absent":
        findings.append("EXIF metadata was absent in the metadata analysis result.")

    if isinstance(fingerprints, Mapping):
        if fingerprints.get("sha256") is not None:
            findings.append("SHA-256 fingerprint result is available.")
        if fingerprints.get("perceptual_hash") is not None:
            findings.append("Perceptual-hash result is available.")

    return findings


def _present_fields(data: Mapping[str, Any], names: tuple[str, ...]) -> str:
    return ", ".join(
        f"{name}={data[name]!r}" for name in names if data.get(name) is not None
    )