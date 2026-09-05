import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from backend.detectors.ai_detector import CapCheckDetector

router = APIRouter()
detector = CapCheckDetector()
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

        result = detector.detect(upload_path)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Image analysis failed.") from exc
    finally:
        image_file.file.close()
        upload_path.unlink(missing_ok=True)
