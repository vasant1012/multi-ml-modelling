import json
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("C:\\Users\\vasant.rajadhvax\\LocalBin_vasant.rajadhyax\\Projects\\multi-ml-modelling")

from backend.config import settings

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

NEGATIVE = "negative"
NEUTRAL = "neutral"
POSITIVE = "positive"


# ------------------------------------------------------------------
# Sentiment Mapping
# ------------------------------------------------------------------

def sentiment_from_rating(rating: int) -> str:
    """
    Convert hotel rating into sentiment label.

    Mapping:
        1,2 -> Negative
        3   -> Neutral
        4,5 -> Positive
    """

    if rating <= 2:
        return NEGATIVE

    if rating == 3:
        return NEUTRAL

    return POSITIVE


# ------------------------------------------------------------------
# Duplicate Normalization
# ------------------------------------------------------------------

def normalize_for_duplicate_check(text: str) -> str:
    """
    Create a normalized version of a review for duplicate detection.

    Example:
        "Great Hotel!"
        "great hotel"
        "Great Hotel."

    All become:

        "great hotel"
    """

    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ------------------------------------------------------------------
# Main Processing Logic
# ------------------------------------------------------------------

def main() -> None:

    # --------------------------------------------------------------
    # Load Data
    # --------------------------------------------------------------

    path = settings.path(settings.data_path)

    print(f"\nLoading dataset from: {path}")

    df = pd.read_csv(path)

    # --------------------------------------------------------------
    # Validate Required Columns
    # --------------------------------------------------------------

    required = {
        settings.text_column,
        settings.rating_column,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Keep only required columns
    df = df[
        [
            settings.text_column,
            settings.rating_column,
        ]
    ].copy()

    df.columns = [
        "review",
        "rating",
    ]

    print(f"Initial records: {len(df):,}")

    # --------------------------------------------------------------
    # Remove Missing Values
    # --------------------------------------------------------------

    df = df.dropna()

    # --------------------------------------------------------------
    # Clean Review Text
    # --------------------------------------------------------------

    df["review"] = (
        df["review"]
        .astype(str)
        .map(lambda x: re.sub(r"\s+", " ", x).strip())
    )

    # Remove empty reviews after cleaning
    df = df[df["review"] != ""]

    # --------------------------------------------------------------
    # Rating Cleanup
    # --------------------------------------------------------------

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    df = df.dropna()

    df = df.query(
        "rating >= 1 and rating <= 5"
    )

    df["rating"] = df["rating"].astype(int)

    # --------------------------------------------------------------
    # Remove Extremely Short Reviews
    # --------------------------------------------------------------

    df = df[df["review"].str.len() > 2]

    # --------------------------------------------------------------
    # Preserve Original Review
    # --------------------------------------------------------------

    df["original_review"] = df["review"]

    # --------------------------------------------------------------
    # Duplicate Detection
    # --------------------------------------------------------------

    df["review_key"] = df["review"].apply(
        normalize_for_duplicate_check
    )

    before_dedup = len(df)

    df = df.drop_duplicates(
        subset="review_key"
    )

    after_dedup = len(df)

    print(
        f"Removed duplicates: "
        f"{before_dedup - after_dedup:,}"
    )

    df = df.drop(columns=["review_key"])

    # --------------------------------------------------------------
    # Generate Metadata
    # --------------------------------------------------------------

    df.insert(
        0,
        "review_id",
        range(1, len(df) + 1)
    )

    df["word_count"] = (
        df["review"]
        .str.split()
        .str.len()
    )

    # --------------------------------------------------------------
    # Create Sentiment Labels
    # --------------------------------------------------------------

    df["sentiment"] = (
        df["rating"]
        .map(sentiment_from_rating)
    )

    # --------------------------------------------------------------
    # Rating Distribution
    # --------------------------------------------------------------

    rating_distribution = (
        df["rating"]
        .value_counts()
        .sort_index()
    )

    print("\nRating Distribution")
    print(rating_distribution)

    # --------------------------------------------------------------
    # Create Output Folder
    # --------------------------------------------------------------

    out = settings.path("data/processed")

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Save Dataset Statistics
    # --------------------------------------------------------------

    stats = {
        "total_records": len(df),
        "average_word_count": round(
            float(df["word_count"].mean()),
            2,
        ),
        "rating_distribution": {
            str(k): int(v)
            for k, v in rating_distribution.items()
        },
        "sentiment_distribution": {
            str(k): int(v)
            for k, v in df["sentiment"]
            .value_counts()
            .items()
        },
    }

    stats_path = out / "dataset_stats.json"

    with open(
        stats_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=4,
        )

    print(
        f"\nSaved dataset statistics:"
        f"\n{stats_path}"
    )

    # --------------------------------------------------------------
    # Train / Validation / Test Split
    #
    # Step 1:
    #     80% Train+Validation
    #     20% Test
    #
    # Step 2:
    #     Split Train into
    #         70% Train
    #         10% Validation
    #
    # Final Ratio:
    #     Train      = 70%
    #     Validation = 10%
    #     Test       = 20%
    # --------------------------------------------------------------

    train, test = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["rating"],
    )

    train, validation = train_test_split(
        train,
        test_size=0.125,
        random_state=42,
        stratify=train["rating"],
    )

    # --------------------------------------------------------------
    # Save Files
    # --------------------------------------------------------------

    train.to_csv(
        out / "train.csv",
        index=False,
    )

    validation.to_csv(
        out / "validation.csv",
        index=False,
    )

    test.to_csv(
        out / "test.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------

    print("\nDataset Split Summary")
    print("-" * 40)
    print(f"Train      : {len(train):,}")
    print(f"Validation : {len(validation):,}")
    print(f"Test       : {len(test):,}")
    print("-" * 40)

    print(
        "\nFiles written to:"
    )
    print(out)


if __name__ == "__main__":
    main()