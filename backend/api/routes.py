import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from backend.core.image_analyzer import analyze_image as analyze_forensic_image
from backend.core.video_analyzer import SUPPORTED_VIDEO_SUFFIXES, analyze_video
from backend.evidence.report import generate_evidence_report

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
