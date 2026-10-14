from fastapi import APIRouter

from app.api.schemas import (
    BatchReviewRequest,
    BatchReviewResponse,
    HealthResponse,
    ModelInfoResponse,
    ReviewRequest,
    ReviewResponse,
)
from app.graph import review_graph

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    return {
        "status": "healthy"
    }


@router.get(
    "/models",
    response_model=ModelInfoResponse,
)
def models():
    return {
        "sentiment_model": "LogisticRegression",
        "rating_model": "LogisticRegression",
    }


@router.post(
    "/analyze",
    response_model=ReviewResponse,
)
def analyze_review(
    request: ReviewRequest
):
    result = review_graph.invoke(
        {
            "review_text": request.review_text
        }
    )

    return result["response"]


@router.post(
    "/analyze-batch",
    response_model=BatchReviewResponse,
)
def analyze_batch(
    request: BatchReviewRequest
):

    results = []

    for review in request.reviews:

        output = review_graph.invoke(
            {
                "review_text": review.review_text
            }
        )

        results.append(
            output["response"]
        )

    return {
        "results": results
    }