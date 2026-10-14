"""Conservative preprocessing that preserves semantic information."""
import html, re, unicodedata
from app.state import ReviewState

def preprocess_text(state: ReviewState) -> dict:
    text = html.unescape(state["original_text"])
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return {"cleaned_text": text, "word_count": len(text.split())}
