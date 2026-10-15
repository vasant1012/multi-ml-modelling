"""
FastAPI routes for hotel-review analysis.
"""

from fastapi import APIRouter

from app.api.schemas import (
    BatchReviewRequest,
    BatchReviewResponse,
    HealthResponse,
    ModelInfoResponse,
    ReviewRequest,
    ReviewResponse,
)
from app.config import settings
from app.graph import review_graph

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> dict:
    """Return basic service health."""

    return {
        "status": "healthy",
    }


@router.get(
    "/models",
    response_model=ModelInfoResponse,
)
def models() -> dict:
    """Return the configured model names."""

    return {
        "sentiment_model": "LogisticRegression",
        "rating_model": "LogisticRegression",
        "emotion_model": settings.emotion_model_name,
    }


@router.post(
    "/analyze",
    response_model=ReviewResponse,
)
def analyze_review(
    request: ReviewRequest,
) -> dict:
    """Analyze one hotel review."""

    result = review_graph.invoke(
        {
            "review_text": request.review_text,
        }
    )

    return result["response"]


@router.post(
    "/analyze-batch",
    response_model=BatchReviewResponse,
)
def analyze_batch(
    request: BatchReviewRequest,
) -> dict:
    """
    Analyze a bounded batch of hotel reviews.

    The batch limit is enforced by the request schema.
    """

    results = []

    for review in request.reviews:
        output = review_graph.invoke(
            {
                "review_text": review.review_text,
            }
        )

        results.append(
            output["response"]
        )

    return {
        "results": results,
    }