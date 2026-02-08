import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, ingestion, query
from app.config import settings
from app.storage.vector_store import vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="RAG Pipeline",
    description="PDF Knowledge Base with Retrieval-Augmented Generation",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingestion.router)
app.include_router(query.router)


@app.on_event("startup")
def startup() -> None:
    """Create data directories and restore vector store from disk."""
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.index_dir, exist_ok=True)

    index_path = os.path.join(settings.index_dir, "chunks.json")
    if os.path.exists(index_path):
        vector_store.load(settings.index_dir)
