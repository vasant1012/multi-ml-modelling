import pandas as pd


def load_data(df: pd.DataFrame) -> pd.DataFrame:
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Example: Basic text cleaning
    df["cleaned_reviews"] = (
        df["review"].str.lower().str.replace(r"[^a-zA-Z\s]", "", regex=True)
    )
    return df


def analyze_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    """Basic sentiment analysis: maps rating to sentiment labels."""

    def rating_to_sentiment(rating):
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"

    data["sentiment"] = data["rating"].apply(rating_to_sentiment)
    return data
