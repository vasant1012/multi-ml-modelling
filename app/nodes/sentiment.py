"""Sentiment prediction using a trained model with a transparent fallback."""
from models.model_registry import load_model
from app.state import ReviewState

POSITIVE = {"great","good","excellent","amazing","clean","friendly","helpful","comfortable","love","perfect","nice","wonderful"}
NEGATIVE = {"bad","dirty","poor","terrible","awful","rude","noisy","broken","worst","disappointed","smell","uncomfortable"}

def _rule_prediction(text: str) -> tuple[str, float]:
    words = set(text.lower().split())
    score = len(words & POSITIVE) - len(words & NEGATIVE)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return label, min(0.55 + abs(score) * 0.08, 0.90)

def predict_sentiment(state: ReviewState) -> dict:
    if "sentiment" not in state["requested_components"]:
        return {}
    model = load_model("models/sentiment/model.joblib")
    text = state["cleaned_text"]
    if model:
        label = str(model.predict([text])[0])
        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities))
        source = "trained_model"
    else:
        label, confidence = _rule_prediction(text)
        source = "rule_fallback"
    return {"sentiment": {"label": label, "confidence": round(confidence, 4), "source": source}}
