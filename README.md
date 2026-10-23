# Hotel Review Intelligence

A modular hotel review analytics platform built using **LangGraph**, **FastAPI**, **scikit-learn**, local **Transformer Models**, and a modern **React Intelligence Workspace**.

The platform converts unstructured hotel reviews into actionable operational insights using multiple AI, ML, and NLP components.

---

# Features

## Core Capabilities

✅ Sentiment Classification

✅ Rating Prediction (1–5 Stars)

✅ Emotion Detection

✅ Aspect-Based Sentiment Analysis (ABSA)

✅ Rule-Based Business Summarization

✅ Cross-Model Reconciliation

✅ Quality Assessment

✅ Batch Review Processing API

✅ LangGraph Workflow Orchestration

✅ FastAPI REST APIs

✅ Automated Test Dashboard

✅ React Intelligence Workspace

✅ Dark / Light Theme Support

✅ Analysis History

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

   ┌───────────┬───────────┬───────────┐
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

├── backend/
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
├── frontend/
│
│   ├── public/
│   ├── src/
│   │   ├── api/
│   │   ├── assets/
│   │   ├── components/
│   │   ├── config/
│   │   ├── hooks/
│   │   ├── pages/
│   │   ├── styles/
│   │   ├── utils/
│   │   ├── App.jsx
│   │   └── main.jsx
│   │
│   ├── package.json
│   ├── vite.config.js
│   └── .env.example
│
├── docs/
│   ├── PROJECT_STATUS.md
│   ├── ARCHITECTURE.md
│   ├── MODEL_RESEARCH.md
│   └── FRONTEND_ROADMAP.md
│
├── data/
│
└── logs/
```

---

# Expected Dataset

Place the hotel review dataset under:

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
pip install -r backend/requirements.txt
```

---

# Data Preparation

Navigate:

```bash
cd Backend
```

Run:

```bash
python -m backend.training.prepare_data
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
python -m backend.training.train_sentiment
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
python -m backend.training.train_rating
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

# Sentiment Model Benchmark Research

A dedicated benchmarking exercise was completed before selecting the production sentiment model.

## Models Evaluated

```text
CountVectorizer + Logistic Regression

TF-IDF + Logistic Regression

TF-IDF + LinearSVC

Sentence Transformer Embeddings
(all-MiniLM-L6-v2)
```

## Benchmark Results

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---------|---------:|---------:|---------:|
| CountVectorizer + Logistic Regression | 0.8458 | 0.6842 | 0.6853 |
| TF-IDF + Logistic Regression | 0.8407 | 0.7226 | 0.7059 |
| TF-IDF + LinearSVC | 0.8707 | 0.6668 | 0.6747 |

## Conclusions

✅ TF-IDF + Logistic Regression provides the best overall balance.

✅ TF-IDF handles Neutral reviews significantly better.

✅ MiniLM embeddings improve calibration but underperform on classification quality.

✅ TF-IDF + LinearSVC improves Accuracy but reduces Balanced Accuracy and Macro F1.

✅ CountVectorizer performs well but does not outperform TF-IDF.

## Production Decision

Selected production model:

```text
TF-IDF + Logistic Regression
```

Sentiment-model experimentation has been frozen for Version 1 so that the project can focus on completing user-facing functionality.

Future research items:

```text
Human Gold Dataset

Mixed Sentiment Framework

SetFit Evaluation

Domain Adapted Transformers
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

# Frontend Setup

Navigate:

```bash
cd Frontend
```

Create environment file:

```bash
cp .env.example .env
```

Example:

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

Install packages:

```bash
npm install
```

Run:

```bash
npm run dev
```

Application:

```text
http://localhost:5173
```

---

# Frontend Status

## Analyze Page ✅ Complete

Route:

```text
/analyze
```

Capabilities:

✅ Review Input Workspace

✅ Sentiment Intelligence Card

✅ Rating Intelligence Card

✅ Emotion Intelligence Card

✅ Quality Assessment

✅ Aspect Intelligence Panel

✅ Evidence Viewer

✅ Summary Workspace

✅ API Health Monitoring

✅ Keyboard Shortcuts

✅ Responsive Layout

✅ Dark Theme

✅ Light Theme

✅ Local Review History Integration

---

## History Page ✅ Complete

Route:

```text
/history
```

Capabilities:

✅ Analysis History

✅ Review Preview

✅ Sentiment Snapshot

✅ Rating Snapshot

✅ Timestamp Tracking

✅ Clear History

✅ Local Storage Persistence

---

## Batch Page 🔄 In Progress

Route:

```text
/batch
```

Current Status:

```text
Placeholder Implemented
```

Planned:

- CSV Upload
- Batch Processing
- Progress Tracking
- Download Results
- Batch Statistics

---

## Insights Dashboard 🔄 In Progress

Route:

```text
/insights
```

Current Status:

```text
Placeholder Implemented
```

Planned:

- Sentiment Trends
- Rating Trends
- Aspect Trends
- Emotion Trends
- Executive Dashboard
- Quality Analytics

---

# API Endpoints

## Health

```http
GET /api/v1/health
```

## Models

```http
GET /api/v1/models
```

## Analyze Single Review

```http
POST /api/v1/analyze
```

## Analyze Multiple Reviews

```http
POST /api/v1/analyze-batch
```

---

# Review Intelligence Output

Each review generates:

```text
Sentiment

Rating

Emotion

ABSA

Summary

Quality Assessment
```

Example:

```json
{
  "sentiment": {
    "label": "negative"
  },

  "rating": {
    "prediction": 2
  },

  "emotion": {
    "dominant": "sadness"
  },

  "aspects": [
    {
      "aspect": "wifi_internet",
      "sentiment": "negative"
    }
  ],

  "summary": {
    "guest_summary": "...",
    "management_summary": "...",
    "executive_summary": "..."
  },

  "quality": {
    "status": "review_required"
  }
}
```

---

# ABSA Categories

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

## Re-run Failed Tests

```bash
python tests/run_tests.py failed
```

---

# Logging

Logs are written to:

```text
Logs/

hotel_review_workflow_YYYY-MM-DD.log
```

Format:

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
| Batch Processing API | ✅ Complete |
| Automated Testing | ✅ Complete |
| Test Dashboard | ✅ Complete |
| Sentiment Benchmark Research | ✅ Complete |
| Production Sentiment Model Selection | ✅ Complete |
| React Frontend Foundation | ✅ Complete |
| Analyze Page | ✅ Complete |
| History Page | ✅ Complete |
| Theme Support | ✅ Complete |
| Batch Analysis UI | 🔄 In Progress |
| Insights Dashboard UI | 🔄 In Progress |
| Sentiment Gold Dataset | ⏳ Future Enhancement |
| Mixed Sentiment Framework | ⏳ Future Enhancement |
| Local LLM Support | ⏳ Future Enhancement |

---

# Current Roadmap

## Phase 1 ✅ Complete

```text
Sentiment Classification
Rating Prediction
Emotion Detection
ABSA
Reconciliation
Summarization
FastAPI
Automated Testing
React Analyze Screen
History Screen
Theme Support
```

---

## Phase 2 🔄 Current Focus

Frontend Completion

```text
Batch Analysis Page

CSV Upload Workflow

Batch Processing Experience

Download Results

Insights Dashboard

Aspect Trends

Emotion Trends

Rating Analytics

Quality Analytics
```

---

## Phase 3 ⏳ Future Enhancements

```text
Sentiment Gold Dataset

Mixed Sentiment Framework

Traveler Persona Detection

Executive Reporting

Local LLM Summaries
```

---

## Phase 4 ⏳ Long-Term Vision

```text
Portfolio Analytics

Multi-Hotel Benchmarking

Trend Forecasting

Recommendation Engine

RAG-Based Review Intelligence
```

---

# Design Principles

The platform follows a modular LangGraph architecture.

Benefits:

- Independent Components
- Easy Testing
- Easy Model Replacement
- Explainable Outputs
- Local Execution
- No Cloud Dependency
- Production-Friendly Architecture
- Future Local LLM Support

Current implementation prioritizes:

```text
Simplicity
Traceability
Reliability
Explainability
Maintainability
```

while establishing a practical foundation for advanced hotel review intelligence and operational analytics.