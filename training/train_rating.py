import json
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.pipeline import Pipeline

sys.path.append("C:\\Users\\vasant.rajadhvax\\LocalBin_vasant.rajadhyax\\Projects\\multi-ml-modelling")
from app.config import settings

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

MODEL_DIR = settings.path("models/rating")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
CLASS_MAP_PATH = MODEL_DIR / "class_mapping.json"


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

print("Loading training data...")

train_df = pd.read_csv(
    settings.path("data/processed/train.csv")
)

valid_df = pd.read_csv(
    settings.path("data/processed/validation.csv")
)

X_train = train_df["review"]
y_train = train_df["rating"]

X_valid = valid_df["review"]
y_valid = valid_df["rating"]

print(f"Train records      : {len(train_df):,}")
print(f"Validation records : {len(valid_df):,}")


# ---------------------------------------------------------
# Model Pipeline
# ---------------------------------------------------------

model = Pipeline(
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


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

print("\nTraining rating model...")

model.fit(
    X_train,
    y_train,
)

print("Training complete")


# ---------------------------------------------------------
# Validate
# ---------------------------------------------------------

predictions = model.predict(
    X_valid
)

accuracy = accuracy_score(
    y_valid,
    predictions,
)

mae = mean_absolute_error(
    y_valid,
    predictions,
)

quadratic_kappa = cohen_kappa_score(
    y_valid,
    predictions,
    weights="quadratic",
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


# ---------------------------------------------------------
# Save Model
# ---------------------------------------------------------

joblib.dump(
    model,
    MODEL_PATH,
)

print(f"Model saved: {MODEL_PATH}")


# ---------------------------------------------------------
# Save Metrics
# ---------------------------------------------------------

metrics = {
    "accuracy": round(float(accuracy), 4),
    "mean_absolute_error": round(float(mae), 4),
    "quadratic_weighted_kappa": round(
        float(quadratic_kappa),
        4,
    ),
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


# ---------------------------------------------------------
# Save Metadata
# ---------------------------------------------------------

metadata = {
    "algorithm": "LogisticRegression",
    "vectorizer": "TFIDF",
    "target": "rating",
    "classes": [1, 2, 3, 4, 5],
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


# ---------------------------------------------------------
# Save Class Mapping
# ---------------------------------------------------------

class_mapping = {
    "1": "Very Negative",
    "2": "Negative",
    "3": "Neutral",
    "4": "Positive",
    "5": "Very Positive",
}

with open(
    CLASS_MAP_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        class_mapping,
        f,
        indent=4,
    )


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

print("\nValidation Results")
print("-" * 50)
print(f"Accuracy          : {accuracy:.4f}")
print(f"MAE               : {mae:.4f}")
print(f"Quadratic Kappa   : {quadratic_kappa:.4f}")
print("-" * 50)

print("\nArtifacts Created")
print(MODEL_PATH)
print(METRICS_PATH)
print(METADATA_PATH)
print(CLASS_MAP_PATH)