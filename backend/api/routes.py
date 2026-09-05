"""HTTP routes exposed by the backend."""

from fastapi import APIRouter


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Report whether the backend is available."""
    return {"status": "healthy"}
