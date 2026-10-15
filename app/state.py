"""
Shared LangGraph state for hotel-review analysis.
"""

from typing import Any, TypedDict


class ReviewState(TypedDict, total=False):
    """
    State shared by all LangGraph nodes.

    total=False allows nodes to add their fields gradually as the
    workflow progresses.
    """

    # Original API input
    review_text: str

    # Preprocessing output
    cleaned_text: str

    # Sentiment-classification output
    sentiment_label: str
    sentiment_confidence: float

    # Rating-prediction output
    predicted_rating: int
    rating_confidence: float

    # Emotion-detection output
    emotions: list[dict[str, Any]]
    dominant_emotion: str
    emotion_confidence: float

    # Reconciliation output
    warnings: list[str]
    consistency_status: str

    # Final API response
    response: dict[str, Any]

    # Validation errors
    errors: list[str]