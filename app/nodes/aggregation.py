from app.state import ReviewState


def aggregate_response(state: ReviewState):
    """
    Convert LangGraph state into
    API response structure.
    """

    return {
        "response": {
            "review_text": state["review_text"],

            "sentiment": {
                "label": state["sentiment_label"],
                "confidence": state["sentiment_confidence"],
            },

            "rating": {
                "prediction": state["predicted_rating"],
                "confidence": state["rating_confidence"],
            },

            "errors": state.get("errors", [])
        }
    }
