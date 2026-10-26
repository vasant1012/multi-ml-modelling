from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import (
    CalibratedClassifierCV,
)
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.append(str(Path.cwd()))  # Add the current working directory to sys.path
from backend.logger import logger

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------

PROJECT_ROOT = Path.cwd()

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "models"
    / "sentiment_benchmark"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ----------------------------------------------------------
# Data
# ----------------------------------------------------------

train_df = pd.read_csv(
    DATA_DIR / "train.csv"
)

valid_df = pd.read_csv(
    DATA_DIR / "validation.csv"
)

test_df = pd.read_csv(
    DATA_DIR / "test.csv"
)

logger.info(
    f"Train shape: {train_df.shape}"
)
logger.info(
    f"Validation shape: {valid_df.shape}"
)
logger.info(
    f"Test shape: {test_df.shape}"
)

TEXT_COLUMN = "review"
LABEL_COLUMN = "sentiment"

X_train = train_df[TEXT_COLUMN]
y_train = train_df[LABEL_COLUMN]

X_valid = valid_df[TEXT_COLUMN]
y_valid = valid_df[LABEL_COLUMN]

X_test = test_df[TEXT_COLUMN]
y_test = test_df[LABEL_COLUMN]

# ----------------------------------------------------------
# Models
# ----------------------------------------------------------

models = {

    "count_lr":

        Pipeline(
            [
                (
                    "vectorizer",
                    CountVectorizer(
                        lowercase=True,
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
        ),

    "tfidf_lr":

        Pipeline(
            [
                (
                    "vectorizer",
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
        ),

    "tfidf_svc":

        Pipeline(
            [
                (
                    "vectorizer",
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

                    CalibratedClassifierCV(
                        estimator=
                            LinearSVC(
                                class_weight="balanced",
                                random_state=42,
                            ),
                        method="sigmoid",
                        cv=5,
                    ),
                ),
            ]
        ),
}

# ----------------------------------------------------------
# Train + Evaluate
# ----------------------------------------------------------

results = []

for name, model in models.items():

    logger.info(
        f"Training {name}"
    )

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    train_time = (
        time.time()
        - start
    )

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    metrics = {

        "model":
            name,

        "accuracy":
            accuracy_score(
                y_test,
                predictions,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_test,
                predictions,
            ),

        "macro_f1":
            f1_score(
                y_test,
                predictions,
                average="macro",
            ),

        "weighted_f1":
            f1_score(
                y_test,
                predictions,
                average="weighted",
            ),

        "log_loss":
            log_loss(
                y_test,
                probabilities,
            ),

        "training_seconds":
            train_time,
    }

    output_path = (
        OUTPUT_DIR
        / name
    )

    output_path.mkdir(
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path
        / "model.joblib",
    )

    report = (
        classification_report(
            y_test,
            predictions,
            output_dict=True,
        )
    )

    with open(
        output_path
        / "classification_report.json",
        "w",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )

    cm = confusion_matrix(
        y_test,
        predictions,
    ).tolist()

    with open(
        output_path
        / "confusion_matrix.json",
        "w",
    ) as f:

        json.dump(
            cm,
            f,
            indent=4,
        )

    results.append(
        metrics
    )

# ----------------------------------------------------------
# Save Benchmark
# ----------------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_DIR
    / "benchmark_results.csv",
    index=False,
)

with open(
    OUTPUT_DIR
    / "benchmark_results.json",
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=4,
    )

logger.info("\nBenchmark Results")

logger.info(
    results_df.sort_values(
        "macro_f1",
        ascending=False,
    )
)