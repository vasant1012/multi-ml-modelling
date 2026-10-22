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
│   ├── public/
│   │
│   ├── src/
│   │
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
├── Data/
│
└── Logs/
```

---

# Expected Dataset

Place the dataset under:

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

# Sentiment Model Research

Current production sentiment model:

```text
TF-IDF + Logistic Regression
```

A benchmark experiment was performed comparing:

```text
TF-IDF + Logistic Regression

vs

Sentence Transformer Embeddings
(all-MiniLM-L6-v2)
```

Current findings:

✅ TF‑IDF performs better on this dataset.

✅ TF‑IDF significantly outperforms MiniLM on Neutral reviews.

✅ MiniLM probabilities are better calibrated.

✅ Further work is focused on:

- Gold Standard Sentiment Dataset
- Mixed Sentiment Framework
- Human Validation
- SetFit Experiments
- Domain Adapted Transformers

The current production model remains:

```text
TF-IDF + Logistic Regression
```

until a new model clearly surpasses it on a human-labeled benchmark set.

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

Open:

```text
http://localhost:5173
```

---

# Frontend Features

## Analyze Page

Route:

```text
/analyze
```

Capabilities:

✅ Review Input Workspace

✅ Sentiment Visualization

✅ Rating Visualization

✅ Emotion Visualization

✅ Quality Assessment

✅ Aspect Intelligence Panel

✅ Evidence Inspection

✅ Summary Workspace

✅ Local Review History Integration

✅ API Health Monitoring

✅ Responsive Layout

✅ Dark / Light Theme Support

---

## History Page

Route:

```text
/history
```

Capabilities:

✅ Local Storage Persistence

✅ Recent Analysis History

✅ Timestamp Tracking

✅ Review Preview

✅ Sentiment Snapshot

✅ Rating Snapshot

✅ Clear History

---

## Batch Page

Status:

```text
Placeholder Implemented
```

Route:

```text
/batch
```

Planned:

- CSV Upload
- Batch Analysis
- Batch Dashboard
- Download Results

---

## Insights Page

Status:

```text
Placeholder Implemented
```

Route:

```text
/insights
```

Planned:

- Aspect Trends
- Rating Trends
- Emotion Trends
- Executive Dashboard
- Business KPIs

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
| React Frontend Foundation | ✅ Complete |
| Analyze Page | ✅ Complete |
| History Page | ✅ Complete |
| Theme Support | ✅ Complete |
| Batch Analysis UI | 🔄 Placeholder |
| Insights Dashboard UI | 🔄 Placeholder |
| Sentiment Gold Dataset | 🔄 Planned |
| Mixed Sentiment Framework | 🔄 Planned |
| Local LLM Support | ⏳ Planned |

---

# Current Roadmap

## Phase 1 ✅

Completed

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
```

---

## Phase 2 🔄

Current Focus

```text
Sentiment Gold Dataset

Human Validation

Mixed Sentiment Framework

Regression Review Pack

SetFit Evaluation
```

---

## Phase 3 🔄

Planned

```text
Batch Review Processing UI

CSV Upload Workflow

Download Results
```

---

## Phase 4 🔄

Planned

```text
Insights Dashboard

Aspect Trends

Emotion Trends

Rating Analytics

Quality Analytics
```

---

## Phase 5 ⏳

Future

```text
Traveler Persona Detection

Local LLM Summaries

Executive Reporting

RAG-based Review Intelligence
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

while building a foundation for advanced hotel review intelligence and operational analytics.