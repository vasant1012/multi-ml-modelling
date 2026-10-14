from typing import TypedDict


class ReviewState(TypedDict, total=False):

    review_text: str

    cleaned_text: str

    sentiment_label: str
    sentiment_confidence: float

    predicted_rating: int
    rating_confidence: float

    response: dict

    errors: list[str]