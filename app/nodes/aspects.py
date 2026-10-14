"""Controlled-vocabulary aspect sentiment with evidence snippets."""
import re
from app.nodes.sentiment import POSITIVE, NEGATIVE
from app.state import ReviewState

ASPECTS = {
    "cleanliness": {"clean","dirty","dust","stain","smell"},
    "staff_service": {"staff","service","reception","manager","housekeeping"},
    "room": {"room","suite"}, "bed_comfort": {"bed","mattress","pillow"},
    "bathroom": {"bathroom","toilet","shower"}, "location": {"location","nearby","distance"},
    "parking": {"parking","garage","park"}, "food": {"food","breakfast","restaurant","dinner"},
    "amenities": {"pool","gym","spa","amenities"}, "noise": {"noise","noisy","loud"},
    "price_value": {"price","cost","expensive","value"}, "internet": {"wifi","internet"},
    "maintenance": {"broken","repair","maintenance"}, "safety": {"safe","unsafe","security"}
}

def analyze_aspects(state: ReviewState) -> dict:
    if "aspects" not in state["requested_components"]:
        return {}
    text = state["cleaned_text"]
    sentences = re.split(r"(?<=[.!?])\s+|\s*;\s*", text)
    results = []
    for aspect, keys in ASPECTS.items():
        evidence = next((s for s in sentences if set(re.findall(r"\b\w+\b", s.lower())) & keys), None)
        if not evidence:
            continue
        words = set(re.findall(r"\b\w+\b", evidence.lower()))
        score = len(words & POSITIVE) - len(words & NEGATIVE)
        sentiment = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        results.append({"aspect": aspect, "sentiment": sentiment, "confidence": round(min(.58 + abs(score)*.1, .9), 4), "evidence": evidence[:220]})
    return {"aspects": results}
