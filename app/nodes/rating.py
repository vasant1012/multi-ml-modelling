"""One-to-five rating prediction with model and fallback paths."""
from models.model_registry import load_model
from app.nodes.sentiment import _rule_prediction
from app.state import ReviewState

def predict_rating(state: ReviewState) -> dict:
    if "rating" not in state["requested_components"]:
        return {}
    model = load_model("models/rating/model.joblib")
    text = state["cleaned_text"]
    if model:
        predicted = int(model.predict([text])[0])
        probs = model.predict_proba([text])[0]
        classes = [int(x) for x in model.classes_]
        probability_map = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
        confidence, source = float(max(probs)), "trained_model"
    else:
        label, confidence = _rule_prediction(text)
        predicted = {"negative": 2, "neutral": 3, "positive": 4}[label]
        probability_map, source = {str(predicted): round(confidence, 4)}, "rule_fallback"
    return {"rating_prediction": {"rating": predicted, "confidence": round(confidence, 4), "probabilities": probability_map, "source": source}}
