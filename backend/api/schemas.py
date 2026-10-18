from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------

class ReviewRequest(BaseModel):
    """Request for analyzing one hotel review."""

    review_text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="Hotel review text to analyze.",
        examples=[
            (
                "The room was clean and comfortable, "
                "but the staff were rude."
            )
        ],
    )


class BatchReviewRequest(BaseModel):
    """Request for analyzing multiple hotel reviews."""

    reviews: list[ReviewRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Hotel reviews to analyze. "
            "A maximum of 100 reviews is accepted per request."
        ),
    )


# ------------------------------------------------------------------
# Component response schemas
# ------------------------------------------------------------------

class SentimentResponse(BaseModel):
    """Overall review sentiment."""

    label: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class RatingResponse(BaseModel):
    """Predicted star rating."""

    prediction: int = Field(
        ge=1,
        le=5,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class EmotionPrediction(BaseModel):
    """One emotion candidate returned by the emotion model."""

    label: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class EmotionResponse(BaseModel):
    """Emotion-detection result."""

    dominant: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    predictions: list[EmotionPrediction] = Field(
        default_factory=list,
    )


class AspectResponse(BaseModel):
    """Sentiment result for one hotel aspect."""

    aspect: str

    sentiment: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence: str


class QualityResponse(BaseModel):
    """Cross-component consistency assessment."""

    status: str

    warnings: list[str] = Field(
        default_factory=list,
    )


# ------------------------------------------------------------------
# Complete response schemas
# ------------------------------------------------------------------

class ReviewResponse(BaseModel):
    """Complete analysis response for one review."""

    review_text: str

    sentiment: SentimentResponse

    rating: RatingResponse

    emotion: EmotionResponse

    aspects: list[AspectResponse] = Field(
        default_factory=list,
    )

    quality: QualityResponse

    errors: list[str] = Field(
        default_factory=list,
    )


class BatchReviewResponse(BaseModel):
    """Complete results for a batch of hotel reviews."""

    results: list[ReviewResponse]


# ------------------------------------------------------------------
# Supporting endpoint schemas
# ------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Application health response."""

    status: str


class ModelInfoResponse(BaseModel):
    """Information about models used by the backend."""

    sentiment_model: str

    rating_model: str

    emotion_model: str