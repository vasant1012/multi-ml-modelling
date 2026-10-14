"""Validate, clean, label and split the raw CSV without data leakage."""
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import settings


def sentiment_from_rating(rating: int) -> str:
    return "negative" if rating <= 2 else "neutral" if rating == 3 else "positive"


def main() -> None:
    path = settings.path(settings.data_path)
    df = pd.read_csv(path)
    required = {settings.text_column, settings.rating_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df[[settings.text_column, settings.rating_column]].copy()
    df.columns = ["review", "rating"]
    df = df.dropna()
    df["review"] = df["review"].astype(str).map(
        lambda x: re.sub(r"\s+", " ", x).strip())
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna().query("rating >= 1 and rating <= 5")
    df["rating"] = df["rating"].astype(int)
    df = df[df["review"].str.len() > 2].drop_duplicates(subset="review")
    df["sentiment"] = df["rating"].map(sentiment_from_rating)
    out = settings.path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    train, test = train_test_split(
        df, test_size=.2, random_state=42, stratify=df["rating"])
    train, valid = train_test_split(
        train, test_size=.125, random_state=42, stratify=train["rating"])
    train.to_csv(out / "train.csv", index=False)
    valid.to_csv(out / "validation.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print({"train": len(train), "validation": len(valid), "test": len(test)})


if __name__ == "__main__":
    main()
