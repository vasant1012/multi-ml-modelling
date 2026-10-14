import sys

sys.path.append("C:\\Users\\vasant.rajadhvax\\LocalBin_vasant.rajadhyax\\Projects\\multi-ml-modelling")

from app.state import ReviewState


def validate_input(state: ReviewState):

    text = state.get("review_text", "")

    errors = []

    if not isinstance(text, str):
        errors.append("Review must be string")

    if not text.strip():
        errors.append("Review cannot be empty")

    return {"errors": errors}