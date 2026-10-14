from typing import List  # noqa: UP035

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    review_text: str = Field(
        ...,
        min_length=1,
        description="Hotel review text"
    )


class BatchReviewRequest(BaseModel):
    reviews: List[ReviewRequest]  # noqa: UP006


class SentimentResponse(BaseModel):
    label: str
    confidence: float


class RatingResponse(BaseModel):
    prediction: int
    confidence: float


class ReviewResponse(BaseModel):
    review_text: str

    sentiment: SentimentResponse

    rating: RatingResponse

    errors: list[str] = []


class BatchReviewResponse(BaseModel):
    results: List[ReviewResponse]  # noqa: UP006


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    sentiment_model: str
    rating_model: str