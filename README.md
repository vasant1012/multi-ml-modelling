# Hotel Review Intelligence

A modular hotel review analytics platform built using **LangGraph**, **FastAPI**, **scikit-learn**, and **local Transformer models**.

The platform analyzes hotel reviews and converts unstructured customer feedback into actionable business insights through multiple AI and ML components.

---

# Features

## Core Capabilities

✅ Sentiment Classification

✅ Rating Prediction (1-5 Stars)

✅ Emotion Detection

✅ Aspect-Based Sentiment Analysis (ABSA)

✅ Review Summarization

✅ Cross-Model Reconciliation

✅ Batch Review Processing

✅ LangGraph Workflow Orchestration

✅ FastAPI REST APIs

✅ Automated Test Dashboard

---

# Architecture Overview

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

   ┌───────────┬───────────┬───────────┬───────────┐
   │           │           │           │
   ▼           ▼           ▼           ▼

Sentiment    Rating     Emotion      ABSA
  Node        Node       Node        Node

   └───────────┴───────────┴───────────┘
                       │
                       ▼

             Reconciliation Node
                       │
                       ▼

              Summarization Node
                       │
                       ▼

               Aggregation Node
                       │
                       ▼

                      END
```

---

# Project Structure

```text
multi-ml-modelling/

├── Backend/
│
│   ├── backend/
│   │
│   │   ├── api/
│   │   ├── nodes/
│   │   ├── subgraphs/
│   │   │
│   │   ├── config.py
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── logger.py
│   │   └── model_registry.py
│   │
│   ├── models/
│   │   ├── sentiment/
│   │   └── rating/
│   │
│   ├── training/
│   │
│   ├── notebooks/
│   │
│   └── tests/
│       ├── run_tests.py
│       ├── test_api.py
│       ├── test_graph.py
│       ├── test_aspects.py
│       ├── test_emotion_graph.py
│       └── test_summarization.py
│
├── Frontend/
│
├── Data/
│
└── Logs/
```

---

# Expected Dataset

Place the hotel review dataset inside:

```text
Data/
```

Example:

```csv
review,rating
"Great location and friendly staff",5
"Dirty room and very noisy",1
```

Required columns:

```text
review
rating
```

Configurable via:

```env
TEXT_COLUMN=review
RATING_COLUMN=rating
```

---

# Environment Variables

Create:

```text
.env
```

Example:

```env
DATA_PATH=Data/hotel_reviews.csv

TEXT_COLUMN=review
RATING_COLUMN=rating

MAX_REVIEW_CHARS=20000

SUMMARY_MIN_WORDS=35

CORS_ORIGINS=http://localhost:5173

EMOTION_MODEL_NAME=j-hartmann/emotion-english-distilroberta-base

EMOTION_CONFIDENCE_THRESHOLD=0.20

EMOTION_MAX_RESULTS=3

HF_LOCAL_FILES_ONLY=true
```

---

# Backend Setup

## Create Virtual Environment

### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Data Preparation

Navigate:

```bash
cd Backend
```

Run:

```bash
python -m training.prepare_data
```

Output:

```text
Data/processed/

├── train.csv
├── validation.csv
├── test.csv
└── dataset_stats.json
```

---

# Training

## Train Sentiment Model

```bash
python -m training.train_sentiment
```

Output:

```text
Backend/models/sentiment/

├── model.joblib
├── metrics.json
├── metadata.json
└── label_mapping.json
```

---

## Train Rating Model

```bash
python -m training.train_rating
```

Output:

```text
Backend/models/rating/

├── model.joblib
├── metrics.json
├── metadata.json
└── class_mapping.json
```

---

## Evaluate Models

```bash
python -m training.evaluate
```

---

# Run FastAPI

From:

```bash
cd Backend
```

Run:

```bash
uvicorn backend.api.main:app --reload
```

Swagger:

```text
http://localhost:8000/docs
```

OpenAPI:

```text
http://localhost:8000/openapi.json
```

---

# API Endpoints

## Health

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

## Models

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

## Analyze Single Review

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
    "confidence": 0.81
  },

  "aspects": [],

  "summary": {
    "guest_summary": "...",
    "management_summary": "...",
    "positive_highlights": [],
    "improvement_opportunities": [],
    "executive_summary": "..."
  },

  "quality": {
    "status": "consistent",
    "warnings": []
  },

  "errors": []
}
```

---

## Analyze Multiple Reviews

```http
POST /api/v1/analyze-batch
```

---

# ABSA Categories

Current aspect taxonomy includes:

```text
room
bed_comfort
bathroom
sleep_quality

staff_service
checkin_checkout
management_resolution

cleanliness
maintenance
amenities

wifi_internet

restaurant
breakfast
bar_lounge

location
transport_access
view_scenery

family_facilities
business_facilities

value_for_money

parking

noise
```

---

# Summarization Outputs

Each review generates:

```text
guest_summary

management_summary

positive_highlights

improvement_opportunities

executive_summary
```

Example:

```json
{
  "executive_summary":
  "Guest experience appears mixed with strong location advantages and Wi-Fi connectivity concerns."
}
```

---

# Testing

## Test Dashboard

Run:

```bash
python tests/run_tests.py
```

Example:

```text
======================================================================
HOTEL REVIEW INTELLIGENCE TEST DASHBOARD
======================================================================

EXECUTION SUMMARY

Suite               Status      Passed    Failed

API                 ✅ PASS      12        0
GRAPH               ✅ PASS      18        0
ABSA                ✅ PASS      15        0
EMOTION             ✅ PASS       7        0
SUMMARY             ✅ PASS       9        0

Total Tests : 61
Passed      : 61
Failed      : 0
Duration    : 4.21 sec
```

---

## Execute Individual Suites

```bash
python tests/run_tests.py api

python tests/run_tests.py graph

python tests/run_tests.py absa

python tests/run_tests.py summary

python tests/run_tests.py emotion
```

---

## Execute Multiple Suites

```bash
python tests/run_tests.py graph summary
```

---

## Re-run Previously Failed Tests

```bash
python tests/run_tests.py failed
```

---

# Logging

The platform uses a centralized logger.

Logs are written to:

```text
Logs/

hotel_review_workflow_YYYY-MM-DD.log
```

Log format:

```text
Timestamp
File
Module
Line Number
Log Level
Message
```

---

# Current Project Status

| Component | Status |
|------------|----------|
| Data Preparation | ✅ Complete |
| Sentiment Classification | ✅ Complete |
| Rating Prediction | ✅ Complete |
| Emotion Detection | ✅ Complete |
| ABSA | ✅ Complete |
| Reconciliation | ✅ Complete |
| Summarization | ✅ Complete |
| FastAPI API | ✅ Complete |
| Batch Processing | ✅ Complete |
| Automated Testing | ✅ Complete |
| Test Dashboard | ✅ Complete |
| React Frontend | ⏳ Next Phase |
| Dashboard Analytics | ⏳ Planned |
| Local LLM Support | ⏳ Planned |

---

# Design Principles

The platform follows a modular LangGraph architecture.

Benefits:

- Independent components
- Easy testing
- Easy model replacement
- Explainable outputs
- Local execution
- No cloud dependency
- Future local LLM support
- Production-friendly structure

Current implementation focuses on simplicity, traceability, maintainability, and real-world hotel review analytics.