from fastapi import APIRouter

from app.config import settings
from app.models.schemas import HealthResponse
from app.storage.vector_store import vector_store

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        mistral_configured=bool(settings.mistral_api_key),
        documents_count=len(vector_store.documents),
        chunks_count=len(vector_store),
        data_dir=settings.data_dir,
        index_dir=settings.index_dir,
    )
