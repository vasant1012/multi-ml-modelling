"""
Shared LangGraph state for hotel-review analysis.
"""

from typing import Any, TypedDict


class ReviewState(TypedDict, total=False):
    # --------------------------------------------------------------
    # Input
    # --------------------------------------------------------------

    review_text: str

    # --------------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------------

    cleaned_text: str

    errors: list[str]

    # --------------------------------------------------------------
    # Sentiment
    # --------------------------------------------------------------

    sentiment_label: str
    sentiment_confidence: float

    # --------------------------------------------------------------
    # Rating
    # --------------------------------------------------------------

    predicted_rating: int
    rating_confidence: float

    # --------------------------------------------------------------
    # Emotion
    # --------------------------------------------------------------

    emotions: list[dict[str, Any]]
    dominant_emotion: str
    emotion_confidence: float

    # --------------------------------------------------------------
    # ABSA
    # --------------------------------------------------------------

    aspects: list[dict[str, Any]]

    # --------------------------------------------------------------
    # Reconciliation
    # --------------------------------------------------------------

    consistency_status: str
    warnings: list[str]

    # --------------------------------------------------------------
    # Summarization
    # --------------------------------------------------------------

    guest_summary: str

    management_summary: str

    positive_highlights: list[str]

    improvement_opportunities: list[str]

    executive_summary: str

    # --------------------------------------------------------------
    # Final API Response
    # --------------------------------------------------------------

    response: dict[str, Any]