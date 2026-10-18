from typing import Any

from backend.state import ReviewState


def aggregate_response(
    state: ReviewState,
) -> dict[str, Any]:
    """
    Convert internal LangGraph state into the public API response.

    Expected response sections:

    - review_text
    - sentiment
    - rating
    - emotion
    - aspects
    - quality
    - errors
    """

    response = {
        "review_text": state.get(
            "review_text",
            "",
        ),

        "sentiment": {
            "label": state.get(
                "sentiment_label",
                "unknown",
            ),
            "confidence": state.get(
                "sentiment_confidence",
                0.0,
            ),
        },

        "rating": {
            "prediction": state.get(
                "predicted_rating",
                3,
            ),
            "confidence": state.get(
                "rating_confidence",
                0.0,
            ),
        },

        "emotion": {
            "dominant": state.get(
                "dominant_emotion",
                "unknown",
            ),
            "confidence": state.get(
                "emotion_confidence",
                0.0,
            ),
            "predictions": state.get(
                "emotions",
                [],
            ),
        },

        "aspects": state.get(
            "aspects",
            [],
        ),

        "quality": {
            "status": state.get(
                "consistency_status",
                "consistent",
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

    return {
        "response": response,
    }