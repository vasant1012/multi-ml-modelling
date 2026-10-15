"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from models.model_registry import warm_up_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models before accepting requests.

    If any required model cannot be loaded, startup fails immediately
    instead of failing during a user request.
    """

    warm_up_models()

    yield


app = FastAPI(
    title="Hotel Review Intelligence",
    version="1.1.0",
    description=(
        "Hotel review analysis using local models, "
        "LangGraph and FastAPI."
    ),
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    router,
    prefix="/api/v1",
)