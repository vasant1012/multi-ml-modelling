"""Input validation node and routing decision."""
from app.config import settings
from app.state import ReviewState

VALID_COMPONENTS = {"sentiment", "rating", "emotion", "aspects", "summary"}

def validate_input(state: ReviewState) -> dict:
    text = state.get("original_text", "")
    errors: list[str] = []
    if not isinstance(text, str) or not text.strip():
        errors.append("Review text must be a non-empty string.")
    elif len(text) > settings.max_review_chars:
        errors.append(f"Review exceeds {settings.max_review_chars} characters.")
    rating = state.get("original_rating")
    if rating is not None and rating not in range(1, 6):
        errors.append("Original rating must be between 1 and 5.")
    requested = state.get("requested_components") or sorted(VALID_COMPONENTS)
    unknown = set(requested) - VALID_COMPONENTS
    if unknown:
        errors.append(f"Unknown components: {sorted(unknown)}")
    return {"errors": errors, "warnings": [], "requested_components": requested}

def route_after_validation(state: ReviewState) -> str:
    """Stop expensive processing when validation failed."""
    return "invalid" if state.get("errors") else "valid"
