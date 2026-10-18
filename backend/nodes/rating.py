from backend.models.model_registry import get_rating_model


def predict_rating(state):

    model = get_rating_model()

    text = state["cleaned_text"]

    prediction = model.predict([text])[0]

    probs = model.predict_proba([text])[0]

    confidence = float(max(probs))

    return {
        "predicted_rating": int(prediction),
        "rating_confidence": round(
            confidence,
            4,
        ),
    }