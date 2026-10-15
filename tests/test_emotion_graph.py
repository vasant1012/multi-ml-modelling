"""
Tests for the emotion-detection node.
"""

from unittest.mock import patch

from app.nodes.emotion import detect_emotions


def fake_emotion_model(text, truncation=True):
    """Return deterministic model-like output."""

    return [
        {
            "label": "joy",
            "score": 0.74,
        },
        {
            "label": "surprise",
            "score": 0.18,
        },
        {
            "label": "neutral",
            "score": 0.08,
        },
    ]


@patch(
    "app.nodes.emotion.get_emotion_model",
    return_value=fake_emotion_model,
)
def test_emotion_detection(mock_get_model):
    state = {
        "cleaned_text": (
            "I loved the room and the staff were wonderful."
        )
    }

    result = detect_emotions(state)

    assert result["dominant_emotion"] == "joy"
    assert result["emotion_confidence"] == 0.74
    assert len(result["emotions"]) >= 1


@patch(
    "app.nodes.emotion.get_emotion_model",
    return_value=fake_emotion_model,
)
def test_emotion_confidence_range(mock_get_model):
    result = detect_emotions(
        {
            "cleaned_text": "Wonderful hotel.",
        }
    )

    for emotion in result["emotions"]:
        assert 0 <= emotion["confidence"] <= 1


def test_empty_text_returns_unknown():
    result = detect_emotions(
        {
            "cleaned_text": "",
        }
    )

    assert result["dominant_emotion"] == "unknown"
    assert result["emotion_confidence"] == 0.0
    assert result["emotions"] == []