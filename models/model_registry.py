"""
Central model registry.

Loads all trained models and caches them in memory.
"""

from functools import lru_cache
from typing import Any

import joblib
from transformers import pipeline

from app.config import settings


@lru_cache(maxsize=1)
def get_sentiment_model():
    model_path = settings.path(
        "models/sentiment/model.joblib"
    )

    return joblib.load(model_path)


@lru_cache(maxsize=1)
def get_rating_model():
    model_path = settings.path(
        "models/rating/model.joblib"
    )

    return joblib.load(model_path)


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

    return joblib.load(model_path)


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

    return joblib.load(model_path)


@lru_cache(maxsize=1)
def get_emotion_model() -> Any:
    """
    Load and cache the pretrained emotion-classification pipeline.

    The first execution downloads model files when
    hf_local_files_only is False. Later executions use the local cache.
    """

    return pipeline(
        task="text-classification",
        model=settings.emotion_model_name,
        tokenizer=settings.emotion_model_name,
        top_k=None,
        truncation=True,
        local_files_only=settings.hf_local_files_only,
        device=-1,
    )


def warm_up_models() -> None:
    """
    Load all models during application startup.

    Warming models at startup prevents the first analyze request from
    paying the model-loading cost.
    """

    get_sentiment_model()
    get_rating_model()
    get_emotion_model()