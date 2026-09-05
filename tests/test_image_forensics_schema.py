"""Tests for the reusable forensic result contract."""

import json

from backend.core.schemas import ForensicResult


def test_successful_result_is_serializable() -> None:
    result = ForensicResult(
        module="example",
        status="success",
        model="example-model",
        evidence=[{"fact": True}],
    )

    payload = json.loads(result.json())
    assert payload["module"] == "example" and payload["model"] == "example-model"
    assert result.error is None


def test_error_result_has_standard_shape() -> None:
    result = ForensicResult(module="example", status="error", error="unavailable")

    assert json.loads(result.json())["error"] == "unavailable"
    assert result.evidence == [] and result.artifacts == []
