import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

sys.path.append(str(Path.cwd()))  # Add the current working directory to sys.path
from backend.config import settings

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MODEL_DIR = settings.path("models/sentiment")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
LABEL_PATH = MODEL_DIR / "label_mapping.json"
METADATA_PATH = MODEL_DIR / "training_metadata.json"


# ------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------

print("Loading training data...")

train_df = pd.read_csv(
    settings.path("data/processed/train.csv")
)

valid_df = pd.read_csv(
    settings.path("data/processed/validation.csv")
)

print(f"Train records      : {len(train_df):,}")
print(f"Validation records : {len(valid_df):,}")


X_train = train_df["review"]
y_train = train_df["sentiment"]

X_valid = valid_df["review"]
y_valid = valid_df["sentiment"]


# ------------------------------------------------------------------
# Build Pipeline
# ------------------------------------------------------------------

model = Pipeline(
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


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

print("\nTraining sentiment model...")

model.fit(
    X_train,
    y_train,
)

print("Training complete")


# ------------------------------------------------------------------
# Validation Predictions
# ------------------------------------------------------------------

predictions = model.predict(
    X_valid
)

accuracy = accuracy_score(
    y_valid,
    predictions,
)

report = classification_report(
    y_valid,
    predictions,
    output_dict=True,
)

cm = confusion_matrix(
    y_valid,
    predictions,
).tolist()


# ------------------------------------------------------------------
# Save Model
# ------------------------------------------------------------------

joblib.dump(
    model,
    MODEL_PATH,
)

print("\nModel saved to:")
print(MODEL_PATH)


# ------------------------------------------------------------------
# Save Metrics
# ------------------------------------------------------------------

metrics = {
    "accuracy": round(float(accuracy), 4),
    "classification_report": report,
    "confusion_matrix": cm,
}

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metrics,
        f,
        indent=4,
    )


# ------------------------------------------------------------------
# Save Label Mapping
# ------------------------------------------------------------------

labels = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

with open(
    LABEL_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        labels,
        f,
        indent=4,
    )


# ------------------------------------------------------------------
# Save Metadata
# ------------------------------------------------------------------

metadata = {
    "algorithm": "LogisticRegression",
    "vectorizer": "TF-IDF",
    "ngram_range": [1, 2],
    "max_features": 50000,
    "train_records": len(train_df),
    "validation_records": len(valid_df),
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
    )


print("\nValidation Accuracy:")
print(round(float(accuracy), 4))

print("\nArtifacts Generated:")
print(MODEL_PATH)
print(METRICS_PATH)
print(LABEL_PATH)
print(METADATA_PATH)