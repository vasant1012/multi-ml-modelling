"""
Central model registry.

Loads all trained models and caches them in memory.
"""

from functools import lru_cache

import joblib

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
