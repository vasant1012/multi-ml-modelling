"""Simple multi-label emotion detector. Replace after obtaining labeled data."""
from app.state import ReviewState

LEXICON = {
    "joy": {"happy","love","amazing","wonderful","delighted","perfect","excellent"},
    "satisfaction": {"satisfied","comfortable","helpful","clean","good","pleasant"},
    "anger": {"angry","rude","unacceptable","worst","furious","terrible"},
    "disappointment": {"disappointed","expected","poor","bad","unfortunately","regret"},
    "sadness": {"sad","unhappy","miserable","depressing"},
    "concern": {"unsafe","worried","concern","fear","scared"},
    "surprise": {"surprised","unexpected","astonished"},
}

def detect_emotions(state: ReviewState) -> dict:
    if "emotion" not in state["requested_components"]:
        return {}
    words = set(state["cleaned_text"].lower().split())
    scores = [(name, len(words & terms)) for name, terms in LEXICON.items()]
    found = [{"label": n, "confidence": round(min(0.55 + count * .12, .91), 4)} for n, count in scores if count]
    if not found:
        found = [{"label": "neutral", "confidence": 0.5}]
    return {"emotions": sorted(found, key=lambda x: x["confidence"], reverse=True)[:3]}
