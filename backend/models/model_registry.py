"""
Central model registry.

Loads all trained models and caches them in memory.
"""

import logging
import os
from functools import lru_cache
from typing import Any

import joblib
from transformers import pipeline

from backend.config import settings

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def _build_sentiment_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words=None,
                    min_df=2,
                    max_df=0.95,
                    max_features=50000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _build_rating_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    min_df=2,
                    max_df=0.95,
                    max_features=50000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _ensure_compatible_model(model: Any, *, model_name: str) -> Any:
    """Fail fast when a stale sklearn pickle is missing expected attributes."""

    if model is None:
        raise ValueError(f"{model_name} model is empty.")

    try:
        if hasattr(model, "predict_proba"):
            model.predict_proba(["sanity check review"])
        return model
    except AttributeError as exc:
        if "multi_class" in str(exc):
            raise RuntimeError(
                f"{model_name} model is incompatible with the active scikit-learn "
                "version; it needs to be retrained."
            ) from exc
        raise


def _train_and_save_sentiment_model() -> Any:
    import pandas as pd

    from backend.training.prepare_data import main as prepare_data_main

    prepare_data_main()

    train_df = pd.read_csv(settings.path("data/processed/train.csv"))
    validation_df = pd.read_csv(settings.path("data/processed/validation.csv"))

    model = _build_sentiment_pipeline()
    model.fit(train_df["review"], train_df["sentiment"])

    validation_predictions = model.predict(validation_df["review"])
    if len(validation_predictions) == 0:
        raise ValueError("Sentiment validation data is empty.")

    model_path = settings.path("models/sentiment/model.joblib")
    joblib.dump(model, model_path)
    return model


def _train_and_save_rating_model() -> Any:
    import pandas as pd

    from backend.training.prepare_data import main as prepare_data_main

    prepare_data_main()

    train_df = pd.read_csv(settings.path("data/processed/train.csv"))
    validation_df = pd.read_csv(settings.path("data/processed/validation.csv"))

    model = _build_rating_pipeline()
    model.fit(train_df["review"], train_df["rating"])

    validation_predictions = model.predict(validation_df["review"])
    if len(validation_predictions) == 0:
        raise ValueError("Rating validation data is empty.")

    model_path = settings.path("models/rating/model.joblib")
    joblib.dump(model, model_path)
    return model


@lru_cache(maxsize=1)
def get_sentiment_model() -> Any:  # noqa: F811
    """Load and cache the trained sentiment model."""

    model_path = settings.path(
        "models/sentiment/model.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Sentiment model not found: {model_path}. "
            "Run training/train_sentiment.py first."
        )

    try:
        model = joblib.load(model_path)
        return _ensure_compatible_model(model, model_name="sentiment")
    except (RuntimeError, AttributeError, OSError, ValueError) as exc:
        logging.warning(
            "Sentiment model is stale or incompatible with the active scikit-learn "
            "version; retraining it now. Reason: %s",
            exc,
        )
        return _train_and_save_sentiment_model()


@lru_cache(maxsize=1)
def get_rating_model() -> Any:  # noqa: F811
    """Load and cache the trained rating model."""

    model_path = settings.path(
        "models/rating/model.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Rating model not found: {model_path}. "
            "Run training/train_rating.py first."
        )

    try:
        model = joblib.load(model_path)
        return _ensure_compatible_model(model, model_name="rating")
    except (RuntimeError, AttributeError, OSError, ValueError) as exc:
        logging.warning(
            "Rating model is stale or incompatible with the active scikit-learn "
            "version; retraining it now. Reason: %s",
            exc,
        )
        return _train_and_save_rating_model()


def _fallback_emotion_model(text: str, truncation: bool = True, **_: Any):
    """Provide a deterministic local emotion classifier when the HF model is unavailable."""

    cleaned = (text or "").lower()
    keyword_map = {
        "joy": [
            "love",
            "loved",
            "amazing",
            "great",
            "wonderful",
            "excellent",
            "happy",
            "good",
            "pleasant",
            "nice",
            "beautiful",
            "perfect",
        ],
        "sadness": [
            "sad",
            "disappointed",
            "unhappy",
            "poor",
            "bad",
            "disappointing",
            "awful",
            "terrible",
            "cry",
            "upset",
            "problem",
        ],
        "anger": [
            "angry",
            "furious",
            "annoyed",
            "rage",
            "hate",
            "ridiculous",
            "outrage",
            "mad",
            "frustrated",
            "irked",
        ],
        "fear": [
            "scared",
            "unsafe",
            "nervous",
            "afraid",
            "danger",
            "risk",
            "panic",
            "worried",
            "stress",
            "threat",
        ],
        "surprise": [
            "surprised",
            "unexpected",
            "wow",
            "amazing",
            "shocked",
            "wowee",
            "sudden",
            "abrupt",
        ],
        "disgust": [
            "disgusting",
            "gross",
            "dirty",
            "filthy",
            "repulsive",
            "nasty",
            "awful",
            "disgusted",
        ],
        "neutral": [
            "room",
            "staff",
            "hotel",
            "breakfast",
            "stay",
            "service",
            "location",
            "check",
            "clean",
            "quiet",
        ],
    }

    scores = {label: 0.0 for label in keyword_map}

    for label, keywords in keyword_map.items():
        for keyword in keywords:
            if keyword in cleaned:
                scores[label] += 0.35

    if max(scores.values()) == 0:
        scores["neutral"] = 0.6

    ranked = [
        {"label": label, "score": round(min(max(score, 0.0), 1.0), 4)}
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if score > 0
    ]

    if not ranked:
        ranked = [{"label": "neutral", "score": 0.6}]

    return ranked


@lru_cache(maxsize=1)
def get_emotion_model() -> Any:
    """
    Load and cache the pretrained emotion-classification pipeline.

    If the configured Hugging Face model cannot be reached offline or in a
    restricted environment, fall back to a lightweight local heuristic so app
    startup and API requests still work without internet access.
    """

    try:
        return pipeline(
            task="text-classification",
            model=settings.emotion_model_name,
            tokenizer=settings.emotion_model_name,
            top_k=None,
            truncation=True,
            local_files_only=settings.hf_local_files_only,
            device=-1,
        )
    except Exception as exc:  # pragma: no cover - exercised via regression tests
        logging.warning(
            "Emotion model unavailable; using local fallback classifier instead. "
            "Reason: %s",
            exc,
        )
        return _fallback_emotion_model


def warm_up_models() -> None:
    """
    Load all models during application startup.

    Warming models at startup prevents the first analyze request from
    paying the model-loading cost.
    """

    get_sentiment_model()
    get_rating_model()
    get_emotion_model()