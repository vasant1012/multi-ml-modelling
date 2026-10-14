import re
import sys

sys.path.append("C:\\Users\\vasant.rajadhvax\\LocalBin_vasant.rajadhyax\\Projects\\multi-ml-modelling")
from app.state import ReviewState


def preprocess_text(state: ReviewState):

    text = state["review_text"]

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return {
        "cleaned_text": text
    }