"""Train a simple five-class rating baseline."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.config import settings


def main():
    df = pd.read_csv(settings.path("data/processed/train.csv"))
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)),
                      ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced"))])
    model.fit(df["review"], df["rating"])
    out = settings.path("models/rating/model.joblib"); out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out); print(f"Saved {out}")
if __name__ == "__main__": main()
