"""
Unit tests for cross-model reconciliation rules.
"""

from backend.nodes.reconciliation import (
    reconcile_results,
)


def test_consistent_positive_output():
    """Aligned positive outputs should not create warnings."""

    state = {
        "sentiment_label": "positive",
        "sentiment_confidence": 0.91,
        "predicted_rating": 5,
        "rating_confidence": 0.76,
        "dominant_emotion": "joy",
        "emotion_confidence": 0.88,
    }

    result = reconcile_results(state)

    assert result["warnings"] == []
    assert (
        result["consistency_status"]
        == "consistent"
    )


def test_positive_sentiment_low_rating_conflict():
    """Positive sentiment and low rating must be flagged."""

    state = {
        "sentiment_label": "positive",
        "sentiment_confidence": 0.90,
        "predicted_rating": 2,
        "rating_confidence": 0.72,
        "dominant_emotion": "joy",
        "emotion_confidence": 0.85,
    }

    result = reconcile_results(state)

    assert (
        "Positive sentiment conflicts with a low "
        "predicted rating."
        in result["warnings"]
    )

    assert (
        result["consistency_status"]
        == "review_required"
    )


def test_negative_sentiment_high_rating_conflict():
    """Negative sentiment and high rating must be flagged."""

    state = {
        "sentiment_label": "negative",
        "sentiment_confidence": 0.87,
        "predicted_rating": 5,
        "rating_confidence": 0.70,
        "dominant_emotion": "sadness",
        "emotion_confidence": 0.81,
    }

    result = reconcile_results(state)

    assert (
        "Negative sentiment conflicts with a high "
        "predicted rating."
        in result["warnings"]
    )


def test_positive_sentiment_negative_emotion_conflict():
    """Positive sentiment and sadness should be flagged."""

    state = {
        "sentiment_label": "positive",
        "sentiment_confidence": 0.90,
        "predicted_rating": 5,
        "rating_confidence": 0.75,
        "dominant_emotion": "sadness",
        "emotion_confidence": 0.85,
    }

    result = reconcile_results(state)

    assert (
        "Positive sentiment conflicts with a negative "
        "dominant emotion."
        in result["warnings"]
    )


def test_low_confidence_predictions_create_warnings():
    """Each low-confidence result should create a warning."""

    state = {
        "sentiment_label": "neutral",
        "sentiment_confidence": 0.40,
        "predicted_rating": 3,
        "rating_confidence": 0.30,
        "dominant_emotion": "neutral",
        "emotion_confidence": 0.35,
    }

    result = reconcile_results(state)

    assert len(result["warnings"]) == 3
    assert (
        result["consistency_status"]
        == "review_required"
    )


def test_missing_outputs_create_warnings():
    """Missing model results must not crash reconciliation."""

    result = reconcile_results({})

    assert len(result["warnings"]) == 3
    assert (
        result["consistency_status"]
        == "review_required"
    )