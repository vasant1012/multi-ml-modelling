"""Shared state passed between LangGraph nodes."""
from typing import Any, TypedDict

class ReviewState(TypedDict, total=False):
    review_id: str
    original_text: str
    original_rating: int | None
    requested_components: list[str]
    cleaned_text: str
    word_count: int
    errors: list[str]
    warnings: list[str]
    sentiment: dict[str, Any]
    rating_prediction: dict[str, Any]
    emotions: list[dict[str, Any]]
    aspects: list[dict[str, Any]]
    summary: dict[str, Any]
    final_result: dict[str, Any]
