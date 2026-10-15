"""
Cross-model reconciliation node.

This node checks whether sentiment, rating, and emotion outputs are
reasonably consistent. It does not change model predictions.

The node returns:

1. A list of human-readable warnings
2. A consistency status:
   - consistent
   - review_required
"""

from typing import Any

from app.state import ReviewState

# Confidence values below these thresholds are considered uncertain.
# These are application-level starting points, not model-training rules.
SENTIMENT_CONFIDENCE_THRESHOLD = 0.60
RATING_CONFIDENCE_THRESHOLD = 0.45
EMOTION_CONFIDENCE_THRESHOLD = 0.50


# Emotions that generally align with positive sentiment.
POSITIVE_EMOTIONS = {
    "joy",
    "satisfaction",
    "surprise",
}

# Emotions that generally align with negative sentiment.
NEGATIVE_EMOTIONS = {
    "anger",
    "disgust",
    "fear",
    "sadness",
    "disappointment",
}

# Neutral is intentionally handled separately.
NEUTRAL_EMOTIONS = {
    "neutral",
}


def reconcile_results(
    state: ReviewState,
) -> dict[str, Any]:
    """
    Check consistency across sentiment, rating, and emotion outputs.

    The function performs four types of checks:

    1. Sentiment versus predicted rating
    2. Sentiment versus dominant emotion
    3. Low-confidence predictions
    4. Missing model outputs

    Predictions are never changed by this node. Any inconsistency is
    reported through the warnings list.

    Args:
        state:
            Current LangGraph state containing model outputs.

    Returns:
        Dictionary containing warnings and consistency status.
    """

    warnings = list(
        state.get("warnings", [])
    )

    sentiment_label = state.get(
        "sentiment_label"
    )

    sentiment_confidence = state.get(
        "sentiment_confidence"
    )

    predicted_rating = state.get(
        "predicted_rating"
    )

    rating_confidence = state.get(
        "rating_confidence"
    )

    dominant_emotion = state.get(
        "dominant_emotion"
    )

    emotion_confidence = state.get(
        "emotion_confidence"
    )

    # ----------------------------------------------------------
    # Check that expected outputs are present
    # ----------------------------------------------------------

    if sentiment_label is None:
        warnings.append(
            "Sentiment prediction is missing."
        )

    if predicted_rating is None:
        warnings.append(
            "Rating prediction is missing."
        )

    if dominant_emotion is None:
        warnings.append(
            "Emotion prediction is missing."
        )

    # ----------------------------------------------------------
    # Sentiment and rating consistency
    # ----------------------------------------------------------

    if (
        sentiment_label == "positive"
        and predicted_rating is not None
        and predicted_rating <= 2
    ):
        warnings.append(
            "Positive sentiment conflicts with a low "
            "predicted rating."
        )

    if (
        sentiment_label == "negative"
        and predicted_rating is not None
        and predicted_rating >= 4
    ):
        warnings.append(
            "Negative sentiment conflicts with a high "
            "predicted rating."
        )

    if (
        sentiment_label == "neutral"
        and predicted_rating is not None
        and predicted_rating in {1, 5}
    ):
        warnings.append(
            "Neutral sentiment conflicts with an extreme "
            "predicted rating."
        )

    # ----------------------------------------------------------
    # Sentiment and emotion consistency
    # ----------------------------------------------------------

    if (
        sentiment_label == "positive"
        and dominant_emotion in NEGATIVE_EMOTIONS
    ):
        warnings.append(
            "Positive sentiment conflicts with a negative "
            "dominant emotion."
        )

    if (
        sentiment_label == "negative"
        and dominant_emotion in POSITIVE_EMOTIONS
    ):
        warnings.append(
            "Negative sentiment conflicts with a positive "
            "dominant emotion."
        )

    if (
        sentiment_label == "neutral"
        and dominant_emotion is not None
        and dominant_emotion not in NEUTRAL_EMOTIONS
        and emotion_confidence is not None
        and emotion_confidence >= 0.80
    ):
        warnings.append(
            "Neutral sentiment conflicts with a strong "
            "non-neutral emotion."
        )

    # ----------------------------------------------------------
    # Confidence checks
    # ----------------------------------------------------------

    if (
        sentiment_confidence is not None
        and sentiment_confidence
        < SENTIMENT_CONFIDENCE_THRESHOLD
    ):
        warnings.append(
            "Sentiment confidence is below the configured "
            "quality threshold."
        )

    if (
        rating_confidence is not None
        and rating_confidence
        < RATING_CONFIDENCE_THRESHOLD
    ):
        warnings.append(
            "Rating confidence is below the configured "
            "quality threshold."
        )

    if (
        emotion_confidence is not None
        and emotion_confidence
        < EMOTION_CONFIDENCE_THRESHOLD
    ):
        warnings.append(
            "Emotion confidence is below the configured "
            "quality threshold."
        )

    # ----------------------------------------------------------
    # Final consistency status
    # ----------------------------------------------------------

    consistency_status = (
        "consistent"
        if not warnings
        else "review_required"
    )

    return {
        "warnings": warnings,
        "consistency_status": consistency_status,
    }