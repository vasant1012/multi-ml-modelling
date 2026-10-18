"""Extractive summarizer. It reuses source sentences and avoids hallucination."""
import re

from backend.config import settings
from backend.nodes.sentiment import NEGATIVE, POSITIVE
from backend.state import ReviewState


def summarize_review(state: ReviewState) -> dict:
    if "summary" not in state["requested_components"]:
        return {}
    text = state["cleaned_text"]
    if state["word_count"] < settings.summary_min_words:
        return {"summary": {"text": text, "generated": False, "reason": "Review is already concise."}}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) == 1:  # Preprocessed datasets sometimes contain no punctuation.
        words = text.split()
        short = " ".join(words[:45]) + ("..." if len(words) > 45 else "")
        return {"summary": {"text": short, "generated": True, "method": "word_limit"}}
    def score(sentence: str) -> int:
        words = set(re.findall(r"\b\w+\b", sentence.lower()))
        return len(words & (POSITIVE | NEGATIVE))
    chosen = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)[:2]
    chosen = [s for _, s in sorted(chosen)]
    return {"summary": {"text": " ".join(chosen), "generated": True, "method": "extractive"}}
