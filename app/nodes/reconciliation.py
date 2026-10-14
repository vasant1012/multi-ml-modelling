"""Detect contradictions between available component outputs."""
from app.state import ReviewState

def reconcile_results(state: ReviewState) -> dict:
    warnings = list(state.get("warnings", []))
    sentiment = state.get("sentiment", {}).get("label")
    predicted = state.get("rating_prediction", {}).get("rating")
    original = state.get("original_rating")
    if sentiment == "positive" and predicted is not None and predicted <= 2:
        warnings.append("Positive sentiment conflicts with a low predicted rating.")
    if sentiment == "negative" and predicted is not None and predicted >= 4:
        warnings.append("Negative sentiment conflicts with a high predicted rating.")
    if original is not None and predicted is not None and abs(original - predicted) >= 2:
        warnings.append("Original and predicted ratings differ by at least two stars.")
    return {"warnings": warnings}
