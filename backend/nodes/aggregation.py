"""
Build final response.
"""

from backend.state import ReviewState


def aggregate_response(
    state: ReviewState,
):

    return {
        "response": {

            "review_text":
                state.get(
                    "review_text",
                    "",
                ),

            "sentiment": {

                "label":
                    state.get(
                        "sentiment_label",
                        "unknown",
                    ),

                "confidence":
                    state.get(
                        "sentiment_confidence",
                        0.0,
                    ),
            },

            "rating": {

                "prediction":
                    state.get(
                        "predicted_rating",
                        3,
                    ),

                "confidence":
                    state.get(
                        "rating_confidence",
                        0.0,
                    ),
            },

            "emotion": {

                "dominant":
                    state.get(
                        "dominant_emotion",
                        "unknown",
                    ),

                "confidence":
                    state.get(
                        "emotion_confidence",
                        0.0,
                    ),

                "predictions":
                    state.get(
                        "emotions",
                        [],
                    ),
            },

            "aspects":
                state.get(
                    "aspects",
                    [],
                ),

            "summary": {

                "guest_summary":
                    state.get(
                        "guest_summary",
                        "",
                    ),

                "management_summary":
                    state.get(
                        "management_summary",
                        "",
                    ),

                "positive_highlights":
                    state.get(
                        "positive_highlights",
                        [],
                    ),

                "improvement_opportunities":
                    state.get(
                        "improvement_opportunities",
                        [],
                    ),

                "executive_summary":
                    state.get(
                        "executive_summary",
                        "",
                    ),
            },

            "quality": {

                "status":
                    state.get(
                        "consistency_status",
                        "consistent",
                    ),

                "warnings":
                    state.get(
                        "warnings",
                        [],
                    ),
            },

            "errors":
                state.get(
                    "errors",
                    [],
                ),
        }
    }