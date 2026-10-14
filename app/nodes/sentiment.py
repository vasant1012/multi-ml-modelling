from models.model_registry import get_sentiment_model


def predict_sentiment(state):

    model = get_sentiment_model()

    text = state["cleaned_text"]

    prediction = model.predict([text])[0]

    probs = model.predict_proba([text])[0]

    confidence = float(max(probs))

    return {
        "sentiment_label": prediction,
        "sentiment_confidence": round(
            confidence,
            4
        ),
    }