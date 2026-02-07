from fastapi import APIRouter

from app.config import settings

router = APIRouter()


@router.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "mistral_configured": bool(settings.mistral_api_key),
        "data_dir": settings.data_dir,
        "index_dir": settings.index_dir,
    }
