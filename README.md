# Hotel Review Intelligence

A modular hotel review analytics platform built using **LangGraph**, **FastAPI**, **scikit-learn**, and **local Transformer models**.

The system analyzes hotel reviews and currently supports:

- Sentiment Classification
- Rating Prediction (1-5 Stars)
- Emotion Detection
- Batch Review Processing
- LangGraph Workflow Orchestration
- REST APIs through FastAPI

The platform is designed to grow incrementally with future support for:

- Aspect-Based Sentiment Analysis (ABSA)
- Review Summarization
- Dashboard Analytics
- React Frontend
- Local LLM Integration

---

## 1. Expected CSV

Place the hotel reviews dataset in:

```text
data/raw/hotel_reviews.csv
```

Expected format:

```csv
review,rating
"Great location and friendly staff",5
"Dirty room and very noisy",1
```

Column names can be configured via environment variables:

```env
TEXT_COLUMN=review
RATING_COLUMN=rating
```

---

## 2. Current Features

### Sentiment Classification

Predicts:

```text
Positive
Neutral
Negative
```

using:

```text
TF-IDF
+
Logistic Regression
```

---

### Rating Prediction

Predicts:

```text
1
2
3
4
5
```

directly from review text.

Uses:

```text
TF-IDF
+
Logistic Regression
```

---

### Emotion Detection

Detects emotions such as:

```text
Joy
Sadness
Anger
Fear
Surprise
Neutral
```

using a locally hosted Transformer model.

Default model:

```text
j-hartmann/emotion-english-distilroberta-base
```

The model is downloaded once and cached locally.

---

### LangGraph Workflow

Current workflow:

```text
Validate Input
        ↓

Preprocess Review
        ↓

 ┌────────────────────┐
 │ Parallel Execution │
 └────────────────────┘

 Sentiment
 Rating
 Emotion

        ↓

 Reconciliation (Planned)

        ↓

 Aggregation

        ↓

 API Response
```

---

## 3. Project Structure

```text
hotel-review-intelligence/

├── app/
│   ├── api/
│   ├── nodes/
│   ├── subgraphs/
│   ├── config.py
│   ├── graph.py
│   └── state.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── labeled/
│
├── models/
│   ├── sentiment/
│   ├── rating/
│   └── model_registry.py
│
├── training/
│   ├── prepare_data.py
│   ├── train_sentiment.py
│   ├── train_rating.py
│   └── evaluate.py
│
├── tests/
│
└── frontend/
```

---

## 4. Environment Variables

Create `.env`

```env
DATA_PATH=data/raw/hotel_reviews.csv

TEXT_COLUMN=review
RATING_COLUMN=rating

MAX_REVIEW_CHARS=20000

CORS_ORIGINS=http://localhost:5173

EMOTION_MODEL_NAME=j-hartmann/emotion-english-distilroberta-base

EMOTION_CONFIDENCE_THRESHOLD=0.20

EMOTION_MAX_RESULTS=3

HF_LOCAL_FILES_ONLY=false
```

---

## 5. Backend Setup

### Create Virtual Environment

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Prepare Data

```bash
python -m training.prepare_data
```

Outputs:

```text
data/processed/

├── train.csv
├── validation.csv
├── test.csv
└── dataset_stats.json
```

---

### Train Sentiment Model

```bash
python -m training.train_sentiment
```

Outputs:

```text
models/sentiment/

├── model.joblib
├── metrics.json
├── metadata.json
└── label_mapping.json
```

---

### Train Rating Model

```bash
python -m training.train_rating
```

Outputs:

```text
models/rating/

├── model.joblib
├── metrics.json
├── metadata.json
└── class_mapping.json
```

---

### Evaluate Models

```bash
python -m training.evaluate
```

---

## 6. Run FastAPI

Start the API server:

```bash
uvicorn app.api.main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

OpenAPI Specification:

```text
http://localhost:8000/openapi.json
```

---

## 7. API Endpoints

### Health Check

```http
GET /api/v1/health
```

Response:

```json
{
  "status": "healthy"
}
```

---

### Model Information

```http
GET /api/v1/models
```

Response:

```json
{
  "sentiment_model": "LogisticRegression",
  "rating_model": "LogisticRegression",
  "emotion_model": "j-hartmann/emotion-english-distilroberta-base"
}
```

---

### Analyze Single Review

```http
POST /api/v1/analyze
```

Request:

```json
{
  "review_text": "Amazing hotel and great staff"
}
```

Response:

```json
{
  "review_text": "Amazing hotel and great staff",
  "sentiment": {
    "label": "positive",
    "confidence": 0.94
  },
  "rating": {
    "prediction": 5,
    "confidence": 0.78
  },
  "emotion": {
    "dominant": "joy",
    "confidence": 0.81,
    "predictions": [
      {
        "label": "joy",
        "confidence": 0.81
      }
    ]
  },
  "errors": [],
  "warnings": []
}
```

---

### Analyze Multiple Reviews

```http
POST /api/v1/analyze-batch
```

Request:

```json
{
  "reviews": [
    {
      "review_text": "Amazing hotel"
    },
    {
      "review_text": "Dirty room"
    }
  ]
}
```

Response:

```json
{
  "results": [
    {
      "review_text": "Amazing hotel",
      "sentiment": {
        "label": "positive",
        "confidence": 0.95
      },
      "rating": {
        "prediction": 5,
        "confidence": 0.82
      },
      "emotion": {
        "dominant": "joy",
        "confidence": 0.80,
        "predictions": [
          {
            "label": "joy",
            "confidence": 0.80
          }
        ]
      },
      "errors": [],
      "warnings": []
    }
  ]
}
```

---

## 8. Run Tests

Run all tests:

```bash
pytest -v
```

or

```bash
pytest -q
```

Current coverage includes:

- Graph Tests
- API Tests
- Batch API Tests
- Emotion Tests
- Model Registry Tests

---

## 9. Current Implementation Status

| Component | Status |
|------------|----------|
| Data Preparation | ✅ Complete |
| Sentiment Model | ✅ Complete |
| Rating Model | ✅ Complete |
| Emotion Detection | ✅ Complete |
| FastAPI | ✅ Complete |
| LangGraph | ✅ Complete |
| Batch Processing | ✅ Complete |
| Automated Tests | ✅ Complete |
| Reconciliation Layer | 🔄 Next |
| ABSA | ⏳ Planned |
| Summarization | ⏳ Planned |
| React Frontend | ⏳ Planned |

---

## 10. LangGraph Architecture

Current implementation:

```text
                 START
                   │
                   ▼
            Validation Node
                   │
                   ▼
          Preprocessing Node
                   │
                   ▼
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼

Sentiment      Rating        Emotion
  Node          Node           Node

    └──────────────┼──────────────┘
                   │
                   ▼
           Aggregation Node
                   │
                   ▼
                  END
```

---

## 11. Model Artifacts

### Sentiment Model

```text
models/sentiment/

├── model.joblib
├── metrics.json
├── metadata.json
└── label_mapping.json
```

### Rating Model

```text
models/rating/

├── model.joblib
├── metrics.json
├── metadata.json
└── class_mapping.json
```

---

## 12. Planned Enhancements

### Phase 1 ✅

```text
Sentiment Classification
Rating Prediction
Emotion Detection
```

### Phase 2 🔄

```text
Reconciliation Layer

Purpose:
- Cross-model consistency checks
- Warning generation
- Confidence validation
```

Example:

```json
{
  "warnings": [
    "Sentiment and predicted rating appear inconsistent."
  ]
}
```

---

### Phase 3 ⏳

```text
Aspect-Based Sentiment Analysis (ABSA)

Review
  ↓
Aspect Extraction
  ↓
Aspect Sentiment
  ↓
Evidence Generation
```

Example:

```json
[
  {
    "aspect": "cleanliness",
    "sentiment": "negative",
    "evidence": "Bathroom was dirty"
  },
  {
    "aspect": "location",
    "sentiment": "positive",
    "evidence": "Near city center"
  }
]
```

---

### Phase 4 ⏳

```text
Review Summarization

Guest Summary
Business Summary
```

Example:

```json
{
  "guest_summary": "Great location and friendly staff.",
  "business_summary": "Guests consistently praise location and staff friendliness."
}
```

---

### Phase 5 ⏳

```text
React Frontend

Features:
- Single Review Analysis
- Batch Review Analysis
- Dashboard Visualization
- Download Results
```

---

## 13. Design Principles

The platform follows a modular architecture where each capability is isolated into a dedicated LangGraph node.

Benefits:

- Easy model replacement
- Easy testing
- Independent component development
- Local execution without paid APIs
- Future LLM integration support

Current implementation favors simplicity, transparency, and maintainability while establishing a strong foundation for advanced review intelligence workflows.