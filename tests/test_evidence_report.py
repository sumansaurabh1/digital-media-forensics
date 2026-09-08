"""Dictionary-only tests for cautious evidence reporting."""

import copy
import json

import pytest

from backend.evidence.report import generate_evidence_report


def _complete(label="human"):
    return {
        "pipeline": "image_forensic_analysis", "status": "success",
        "metadata": {"module": "metadata", "status": "success", "label": "metadata_absent"},
        "fingerprints": {"sha256": {"value": "abc"}, "perceptual_hash": {"value": "123"}},
        "traceability": {"module": "google_cloud_vision_web_detection", "status": "success", "pages_with_matching_images": [{"url": "https://example.test"}]},
        "ai_detection": {"module": "ai_detector", "model": "test-model", "status": "success", "label": label,
                         "human_score": 0.8, "ai_score": 0.2, "confidence": "medium", "device": "cpu", "error": None},
        "manipulation": {"module": "manipulation", "status": "success", "label": "weak_signals", "score": 0.25,
                           "confidence": "low", "evidence": [{"type": "signal"}]},
        "errors": [],
    }


@pytest.mark.parametrize("value", [None, {}, {"status": "success"}, {"metadata": None, "ai_detection": None, "manipulation": None, "fingerprints": None, "errors": None}, {"metadata": {}, "ai_detection": {}, "manipulation": {}, "fingerprints": {}, "errors": []}, ["malformed"]])
def test_report_handles_missing_none_empty_and_malformed_inputs(value):
    json.dumps(generate_evidence_report(value))


def test_complete_result_preserves_raw_results_and_does_not_mutate_input():
    result = _complete()
    original = copy.deepcopy(result)
    report = generate_evidence_report(result)
    assert result == original
    assert report["ai_detection"] == result["ai_detection"]
    assert report["manipulation"] == result["manipulation"]
    assert report["fingerprints"] == result["fingerprints"]
    assert report["traceability"] == result["traceability"]
    assert report["metadata_summary"] == result["metadata"]
    assert report["pipeline_errors"] == result["errors"]
    assert "EXIF metadata was absent" in " ".join(report["supporting_evidence"]["observed"])
    assert "Source/timeline was not established by the available analysis." in report["warnings_limitations"]


def test_likely_ai_and_human_results_use_cautious_wording():
    ai_report = generate_evidence_report(_complete("ai"))
    human_report = generate_evidence_report(_complete("human"))
    assert "likely AI-generated" in " ".join(ai_report["supporting_evidence"]["interpretation"])
    assert "likely human-origin" in " ".join(human_report["supporting_evidence"]["interpretation"])
    assert "not, by itself, a definitive" in " ".join(ai_report["supporting_evidence"]["limitations"])


def test_manipulation_is_reported_as_signal_not_probability():
    report = generate_evidence_report(_complete())
    text = " ".join(report["supporting_evidence"]["interpretation"] + report["supporting_evidence"]["limitations"])
    assert "inconsistency signals" in text
    assert "probability of manipulation" not in text.lower()
    assert "definitively edited" in text


@pytest.mark.parametrize("status", ["partial", "failure"])
def test_partial_and_failure_results_expose_incomplete_investigation(status):
    result = _complete()
    result["status"] = status
    result["errors"] = [{"module": "ai_detection", "error": "unavailable"}]
    report = generate_evidence_report(result)
    assert "incomplete" in report["overall_assessment"].lower()
    assert report["pipeline_errors"] == result["errors"]


def test_report_has_no_unsupported_certainty_claims_and_is_json_serializable():
    report = generate_evidence_report(_complete("ai"))
    rendered = json.dumps(report).lower()
    for forbidden in ("guaranteed", "definitively ai-generated", "definitively authentic", "100% authentic", "100% ai-generated", "probability of manipulation"):
        assert forbidden not in rendered
    for forbidden in ("is the original source", "first-ever upload", "exact creator", "establishes ownership"):
        assert forbidden not in rendered
