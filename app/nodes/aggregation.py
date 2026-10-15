"""
Aggregate LangGraph outputs into the API response.
"""

from typing import Any

from app.state import ReviewState


def aggregate_response(
    state: ReviewState,
) -> dict[str, Any]:
    """
    Convert internal graph state into a stable API response.

    The API receives nested output instead of raw LangGraph fields.
    This contract can be extended later with aspects and summaries.
    """

    return {
        "response": {
            "review_text": state["review_text"],
            "sentiment": {
                "label": state[
                    "sentiment_label"
                ],
                "confidence": state[
                    "sentiment_confidence"
                ],
            },
            "rating": {
                "prediction": state[
                    "predicted_rating"
                ],
                "confidence": state[
                    "rating_confidence"
                ],
            },
            "emotion": {
                "dominant": state[
                    "dominant_emotion"
                ],
                "confidence": state[
                    "emotion_confidence"
                ],
                "predictions": state.get(
                    "emotions",
                    [],
                ),
            },
            "quality": {
                "status": state.get(
                    "consistency_status",
                    "unknown",
                ),
                "warnings": state.get(
                    "warnings",
                    [],
                ),
            },
            "errors": state.get(
                "errors",
                [],
            ),
        }
    }