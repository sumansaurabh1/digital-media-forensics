import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from backend.core.image_analyzer import analyze_image as analyze_forensic_image
from backend.core.video_analyzer import SUPPORTED_VIDEO_SUFFIXES, analyze_video
from backend.evidence.report import generate_evidence_report
from backend.core.traceability import analyze_traceability
from backend.detectors.morph_detector.detector import analyze_morph

router = APIRouter()
UPLOAD_DIR = Path("data/uploads")

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@router.get("/health")
def health():
    return {"status": "healthy"}


@router.post("/analyze/image")
async def analyze_image(image_file: UploadFile = File(..., alias="image")):
    suffix = Path(image_file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Supported image types: PNG, JPG, JPEG, WEBP.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"

    try:
        with upload_path.open("wb") as output:
            shutil.copyfileobj(image_file.file, output)

        with Image.open(upload_path) as image:
            image.verify()

        result = analyze_forensic_image(upload_path)
        result["traceability"] = analyze_traceability(upload_path)
        report = generate_evidence_report(result)
        result["evidence_report"] = report
        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Image analysis failed.") from exc
    finally:
        image_file.file.close()
        upload_path.unlink(missing_ok=True)


@router.post("/analyze/video")
async def analyze_video_upload(video_file: UploadFile = File(..., alias="video")):
    suffix = Path(video_file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Supported video types: MP4, MOV, AVI, WEBM.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    try:
        with upload_path.open("wb") as output:
            shutil.copyfileobj(video_file.file, output)
        return analyze_video(upload_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Video analysis failed.") from exc
    finally:
        video_file.file.close()
        upload_path.unlink(missing_ok=True)


@router.post("/analyze/morph")
async def analyze_morph_upload(
    target: UploadFile = File(...), reference_a: UploadFile = File(...), reference_b: UploadFile | None = File(None)
):
    uploads = [("target", target), ("reference_a", reference_a)] + ([] if reference_b is None else [("reference_b", reference_b)])
    paths: dict[str, Path] = {}
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for field, upload in uploads:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                raise HTTPException(status_code=400, detail=f"{field}: supported image types are PNG, JPG, JPEG, WEBP.")
            path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
            paths[field] = path
            with path.open("wb") as output:
                shutil.copyfileobj(upload.file, output)
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"{field}: invalid image content.") from exc
        return analyze_morph(paths["target"], paths["reference_a"], paths.get("reference_b"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Morph analysis failed.") from exc
    finally:
        for _, upload in uploads:
            try:
                await upload.close()
            except Exception:
                pass
        for path in paths.values():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
