"""Evaluate both trained models on the untouched test split."""

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    mean_absolute_error,
)

from backend.config import settings


def main():
    df = pd.read_csv(settings.path("data/processed/test.csv"))
    sentiment = joblib.load(settings.path("models/sentiment/model.joblib"))
    rating = joblib.load(settings.path("models/rating/model.joblib"))
    sent_pred, rating_pred = sentiment.predict(df.review), rating.predict(df.review)
    metrics = {
        "sentiment_accuracy": accuracy_score(df.sentiment, sent_pred),
        "sentiment_report": classification_report(
            df.sentiment, sent_pred, output_dict=True
        ),
        "rating_accuracy": accuracy_score(df.rating, rating_pred),
        "rating_mae": mean_absolute_error(df.rating, rating_pred),
        "rating_quadratic_kappa": cohen_kappa_score(
            df.rating, rating_pred, weights="quadratic"
        ),
    }
    out = settings.path("models/evaluation.json")
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
