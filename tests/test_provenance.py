import json
import sys
from types import SimpleNamespace

from backend.detectors.provenance.analyzer import analyze, _extract_evidence


def test_extract_evidence_keeps_expected_c2pa_fields():
    manifest = {
        "active_manifest": {"title": "image.png"},
        "manifests": {"count": 1},
        "validation_status": [{"code": "signingCredential.untrusted"}],
        "validation_results": {"activeManifest": {"success": [{"code": "claimSignature.validated"}]}},
    }

    evidence = _extract_evidence(manifest)

    assert evidence[0] == {"type": "c2pa", "state": "present"}
    assert [item["name"] for item in evidence[1:]] == [
        "active_manifest",
        "manifests",
        "validation_status",
        "validation_results",
    ]


def test_analyze_reports_provenance_absent(tmp_path, monkeypatch):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    class Reader:
        def __init__(self, path):
            self.path = path

        def is_embedded(self):
            return False

    monkeypatch.setitem(sys.modules, "c2pa", SimpleNamespace(Reader=Reader))

    result = analyze(image)

    assert result.module == "provenance"
    assert result.status == "success"
    assert result.label == "provenance_absent"
    assert result.evidence == [{"type": "c2pa", "state": "absent"}]


def test_analyze_extracts_embedded_c2pa(tmp_path, monkeypatch):
    image = tmp_path / "test.jpg"
    image.write_bytes(b"test")

    manifest = {
        "active_manifest": {"title": "image.png"},
        "validation_status": [{"code": "signingCredential.untrusted"}],
    }

    class Reader:
        def __init__(self, path):
            self.path = path

        def is_embedded(self):
            return True

        def json(self):
            return json.dumps(manifest)

    monkeypatch.setitem(sys.modules, "c2pa", SimpleNamespace(Reader=Reader))

    result = analyze(image)

    assert result.module == "provenance"
    assert result.status == "success"
    assert result.label == "provenance_present"
    assert result.evidence[0] == {"type": "c2pa", "state": "present"}
    assert result.evidence[1]["name"] == "active_manifest"
    assert result.evidence[2]["name"] == "validation_status"


def test_analyze_handles_missing_image(tmp_path):
    result = analyze(tmp_path / "missing.jpg")

    assert result.module == "provenance"
    assert result.status == "error"
    assert "Image path does not exist" in result.error
