from typing import Any, TypedDict


class ReviewState(TypedDict, total=False):
    """
    State shared by all hotel-review workflow nodes.

    total=False means nodes are not required to return every field.
    For example, the rating node returns only rating-related values.
    """

    # --------------------------------------------------------------
    # Input
    # --------------------------------------------------------------

    review_text: str

    # --------------------------------------------------------------
    # Validation and preprocessing
    # --------------------------------------------------------------

    cleaned_text: str

    errors: list[str]

    # --------------------------------------------------------------
    # Sentiment classification
    # --------------------------------------------------------------

    sentiment_label: str

    sentiment_confidence: float

    # --------------------------------------------------------------
    # Rating prediction
    # --------------------------------------------------------------

    predicted_rating: int

    rating_confidence: float

    # --------------------------------------------------------------
    # Emotion detection
    # --------------------------------------------------------------

    emotions: list[dict[str, Any]]

    dominant_emotion: str

    emotion_confidence: float

    # --------------------------------------------------------------
    # Aspect-Based Sentiment Analysis
    # --------------------------------------------------------------

    aspects: list[dict[str, Any]]

    # --------------------------------------------------------------
    # Reconciliation and quality assessment
    # --------------------------------------------------------------

    consistency_status: str

    warnings: list[str]

    # --------------------------------------------------------------
    # Final API response
    # --------------------------------------------------------------

    response: dict[str, Any]