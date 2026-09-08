from pathlib import Path
import hashlib
import json
import subprocess
from typing import Any

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def exif_metadata(path: Path) -> dict[str, Any]:
    try:
        r = subprocess.run(["EXIF.exe", str(path)], capture_output=True, text=True, timeout=15)
        return {"raw": r.stdout.strip(), "error": r.stderr.strip() or None}
    except Exception as exc:
        return {"raw": None, "error": str(exc)}

def analyze_traceability(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    result = {
        "sha256": sha256(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "exif": exif_metadata(path),
        "c2pa": None,
        "provenance_status": "metadata_only",
    }

    try:
        from c2pa import Reader
        with Reader(str(path)) as reader:
            result["c2pa"] = json.loads(reader.json())
            result["provenance_status"] = "provenance_present"
    except Exception:
        pass

    return result
