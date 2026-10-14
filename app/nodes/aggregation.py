"""Create the stable response contract returned by the API."""
from app.state import ReviewState

def aggregate_results(state: ReviewState) -> dict:
    result = {
        "review_id": state.get("review_id"),
        "original_text": state.get("original_text"),
        "original_rating": state.get("original_rating"),
        "sentiment": state.get("sentiment"),
        "rating_prediction": state.get("rating_prediction"),
        "emotions": state.get("emotions", []),
        "aspects": state.get("aspects", []),
        "summary": state.get("summary"),
        "warnings": state.get("warnings", []),
        "errors": state.get("errors", []),
    }
    return {"final_result": result}
