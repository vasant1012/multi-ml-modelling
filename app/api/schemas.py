"""
Pydantic request and response schemas for the review API.
"""

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    """Request for a single hotel review."""

    review_text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="Hotel review text to analyze.",
    )


class BatchReviewRequest(BaseModel):
    """Request containing multiple hotel reviews."""

    reviews: list[ReviewRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
    )


class SentimentResponse(BaseModel):
    """Sentiment-classification response."""

    label: str
    confidence: float = Field(
        ge=0,
        le=1,
    )


class RatingResponse(BaseModel):
    """One-to-five rating-prediction response."""

    prediction: int = Field(
        ge=1,
        le=5,
    )

    confidence: float = Field(
        ge=0,
        le=1,
    )


class EmotionPrediction(BaseModel):
    """One possible emotion and its confidence."""

    label: str

    confidence: float = Field(
        ge=0,
        le=1,
    )


class EmotionResponse(BaseModel):
    """Emotion-detection response."""

    dominant: str

    confidence: float = Field(
        ge=0,
        le=1,
    )

    predictions: list[
        EmotionPrediction
    ] = Field(
        default_factory=list
    )


class QualityResponse(BaseModel):
    """Cross-model consistency result."""

    status: str

    warnings: list[str] = Field(
        default_factory=list
    )


class ReviewResponse(BaseModel):
    """Complete analysis response for one review."""

    review_text: str
    sentiment: SentimentResponse
    rating: RatingResponse
    emotion: EmotionResponse
    quality: QualityResponse

    errors: list[str] = Field(
        default_factory=list
    )


class BatchReviewResponse(BaseModel):
    """Analysis output for multiple reviews."""

    results: list[ReviewResponse]


class HealthResponse(BaseModel):
    """Service health response."""

    status: str


class ModelInfoResponse(BaseModel):
    """Names of models used by the service."""

    sentiment_model: str
    rating_model: str
    emotion_model: str