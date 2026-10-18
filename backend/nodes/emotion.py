"""
Local emotion-detection LangGraph node.

The node uses a pretrained Hugging Face text-classification model.
It returns a dominant emotion plus a filtered list of predictions.
"""

from typing import Any

from backend.config import settings
from backend.state import ReviewState
from models.model_registry import get_emotion_model

# This map gives the API stable, lowercase emotion names.
# Unknown model labels are converted to lowercase automatically.
EMOTION_LABEL_MAP = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
}


def normalize_emotion_label(label: str) -> str:
    """
    Normalize a model label for the public API.

    For example:
        "JOY" -> "joy"
        "LABEL_0" -> "label_0"
    """

    normalized = label.strip().lower()

    return EMOTION_LABEL_MAP.get(
        normalized,
        normalized,
    )


def _unwrap_pipeline_output(
    raw_output: Any,
) -> list[dict[str, Any]]:
    """
    Normalize possible Transformers pipeline output shapes.

    A single text can produce either:

        [{"label": "joy", "score": 0.9}, ...]

    or:

        [[{"label": "joy", "score": 0.9}, ...]]

    depending on the installed Transformers version and invocation.
    """

    if not isinstance(raw_output, list):
        raise ValueError(  # noqa: TRY004
            "Emotion model returned an unexpected output type."
        )

    if (
        raw_output
        and isinstance(raw_output[0], list)
    ):
        raw_output = raw_output[0]

    return raw_output


def detect_emotions(
    state: ReviewState,
) -> dict[str, Any]:
    """
    Detect emotions from the cleaned review text.

    The model returns a score for each supported emotion. Predictions
    are sorted by confidence and filtered using application settings.

    At least the highest-scoring prediction is returned, even when it
    falls below the configured threshold.
    """

    text = state.get("cleaned_text", "").strip()

    if not text:
        return {
            "emotions": [],
            "dominant_emotion": "unknown",
            "emotion_confidence": 0.0,
        }

    classifier = get_emotion_model()

    raw_output = classifier(
        text,
        truncation=True,
    )

    predictions = _unwrap_pipeline_output(
        raw_output
    )

    normalized_predictions = []

    for prediction in predictions:
        label = normalize_emotion_label(
            str(prediction["label"])
        )

        score = float(
            prediction["score"]
        )

        normalized_predictions.append(
            {
                "label": label,
                "confidence": round(score, 4),
            }
        )

    normalized_predictions.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    filtered_predictions = [
        item
        for item in normalized_predictions
        if item["confidence"]
        >= settings.emotion_confidence_threshold
    ]

    # Always provide the strongest prediction.
    if (
        not filtered_predictions
        and normalized_predictions
    ):
        filtered_predictions = [
            normalized_predictions[0]
        ]

    filtered_predictions = filtered_predictions[
        : settings.emotion_max_results
    ]

    if not filtered_predictions:
        return {
            "emotions": [],
            "dominant_emotion": "unknown",
            "emotion_confidence": 0.0,
        }

    dominant = filtered_predictions[0]

    return {
        "emotions": filtered_predictions,
        "dominant_emotion": dominant["label"],
        "emotion_confidence": dominant["confidence"],
    }