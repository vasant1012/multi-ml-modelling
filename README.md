# Hotel Review Intelligence

A modular hotel review analytics platform built using **LangGraph**, **FastAPI**, **scikit-learn**, local **Transformer models**, and a modern **React Intelligence Workspace**.

The platform converts unstructured hotel reviews into explainable sentiment, rating, emotion, aspect, quality, and management-level insights for both individual reviews and review portfolios.

## Version 1 Status

**Development Status: ✅ Complete**

Version 1 now provides an end-to-end experience covering:

- Individual hotel review analysis
- Historical analysis restoration
- Batch CSV analysis
- Downloadable batch results
- Portfolio-level analytics
- Interactive visualizations
- Deterministic executive insights

The Version 1 backend and planned frontend feature scope are complete.

Production hardening, dependency upgrades, frontend automated testing, and additional advanced analytics remain separate future activities.

---

# Features

## Core Intelligence Capabilities

- ✅ Sentiment Classification
- ✅ Rating Prediction (1–5 Stars)
- ✅ Emotion Detection
- ✅ Aspect-Based Sentiment Analysis (ABSA)
- ✅ Cross-Model Reconciliation
- ✅ Rule-Based Business Summarization
- ✅ Quality Assessment
- ✅ LangGraph Workflow Orchestration
- ✅ FastAPI REST APIs
- ✅ Single Review Analysis
- ✅ Batch Review Processing
- ✅ Automated Backend Test Dashboard

## Frontend Capabilities

- ✅ React Intelligence Workspace
- ✅ Analyze Workspace
- ✅ Historical Analysis
- ✅ Historical Result Rehydration
- ✅ CSV Batch Analysis
- ✅ Batch Validation and Preview
- ✅ Batch Results Table
- ✅ Downloadable CSV Results
- ✅ Portfolio Insights Dashboard
- ✅ Sentiment Visualization
- ✅ Rating Visualization
- ✅ Emotion Visualization
- ✅ Aspect Intelligence Visualization
- ✅ Executive Summary
- ✅ Responsive Landscape-First Layout
- ✅ Dark Theme
- ✅ Light Theme
- ✅ Browser-Based Persistence

---

# Version 1 User Experience

The frontend is organized around four primary workflows:

```text
Analyze
   │
   ├── Single Review Intelligence
   └── Save to History

History
   │
   └── Restore Previous Analysis
              │
              ▼
           Analyze

Batch
   │
   ├── Upload CSV
   ├── Validate
   ├── Preview
   ├── Analyze
   ├── Review Results
   └── Download CSV
              │
              ▼
        Persist Latest Batch

Insights
   │
   ├── KPI Summary
   ├── Sentiment Analytics
   ├── Rating Analytics
   ├── Emotion Analytics
   ├── Aspect Intelligence
   └── Executive Summary
```

---

# Architecture Overview

## Backend Intelligence Workflow

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
   │           │           │           │
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

## Application Flow

```text
React Frontend
      │
      ▼
FastAPI REST API
      │
      ▼
LangGraph Workflow
      │
      ├── Sentiment
      ├── Rating
      ├── Emotion
      └── ABSA
      │
      ▼
Reconciliation
      │
      ▼
Summarization
      │
      ▼
Aggregated Intelligence Response
      │
      ▼
React Visualization / Persistence
```

---

# Project Structure

```text
multi-ml-modelling/

├── backend/
│   │
│   ├── backend/
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
│   ├── training/
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
│   │
│   ├── public/
│   │
│   ├── src/
│   │   ├── api/
│   │   ├── assets/
│   │   ├── components/
│   │   │   ├── aspects/
│   │   │   ├── batch/
│   │   │   ├── common/
│   │   │   ├── insights/
│   │   │   ├── layout/
│   │   │   ├── metrics/
│   │   │   ├── review/
│   │   │   └── summary/
│   │   │
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
├── logs/
│
└── README.md
```

---

# Technology Stack

## Backend

- Python
- FastAPI
- LangGraph
- scikit-learn
- Transformers
- PyTorch
- Pandas
- Pytest

## Frontend

- React
- Vite
- React Router
- Axios
- Lucide Icons
- Papa Parse
- Recharts
- Responsive CSS
- Local Storage
- Session Storage

---

# Expected Dataset

Place the hotel review dataset under:

```text
data/
```

Example:

```csv
review,rating
"Great location and friendly staff",5
"Dirty room and very noisy",1
```

Required training-data columns:

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

# Batch CSV Format

The Version 1 Batch Analysis frontend accepts CSV input containing:

```text
review_text
```

Example:

```csv
review_text
"Great location and friendly staff"
"Dirty room and very noisy"
"Excellent service, clean room and comfortable bed"
```

The Batch workflow supports:

- CSV file selection
- Drag-and-drop upload
- Quoted CSV fields
- Multiline CSV handling
- Empty-row handling
- Review-length validation
- File validation
- Review preview
- Batch processing
- Results inspection
- CSV results download

The batch API currently accepts up to **100 reviews per request**.

---

# Environment Variables

Create:

```text
.env
```

Example:

```env
DATA_PATH=data/hotel_reviews.csv

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

### Linux / macOS / WSL

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

# Data Preparation

From the project root, navigate to:

```bash
cd backend
```

Run:

```bash
python -m backend.training.prepare_data
```

Expected processed data:

```text
data/processed/

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

Example model artifacts:

```text
backend/models/sentiment/

├── model.joblib
├── metrics.json
├── metadata.json
└── label_mapping.json
```

## Train Rating Model

```bash
python -m backend.training.train_rating
```

Example model artifacts:

```text
backend/models/rating/

├── model.joblib
├── metrics.json
├── metadata.json
└── class_mapping.json
```

## Evaluate Models

```bash
python -m training.evaluate
```

---

# Sentiment Model Benchmark Research

A dedicated benchmarking exercise was completed before selecting the Version 1 production sentiment model.

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
|---|---:|---:|---:|
| CountVectorizer + Logistic Regression | 0.8458 | 0.6842 | 0.6853 |
| TF-IDF + Logistic Regression | 0.8407 | 0.7226 | 0.7059 |
| TF-IDF + LinearSVC | 0.8707 | 0.6668 | 0.6747 |

## Conclusions

- ✅ TF-IDF + Logistic Regression provides the best overall balance.
- ✅ TF-IDF handles Neutral reviews significantly better.
- ✅ MiniLM embeddings improve calibration but underperform on classification quality.
- ✅ TF-IDF + LinearSVC improves Accuracy but reduces Balanced Accuracy and Macro F1.
- ✅ CountVectorizer performs well but does not outperform TF-IDF overall.

## Production Decision

Selected Version 1 sentiment model:

```text
TF-IDF + Logistic Regression
```

Sentiment-model experimentation is frozen for Version 1.

The following research items are intentionally deferred:

- Human Gold Dataset
- Mixed Sentiment Framework
- SetFit Evaluation
- Domain-Adapted Transformers

---

# Run FastAPI

Navigate to:

```bash
cd backend
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

Navigate to:

```bash
cd frontend
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

## Frontend Build Validation

The Version 1 frontend has been successfully compiled using:

```bash
npm run build
```

The current implementation builds successfully under Vite.

Production hardening and bundle optimization are intentionally outside the current development-ready Version 1 scope.

---

# Frontend Status

## Analyze Page ✅ Complete

Route:

```text
/analyze
```

Capabilities:

- ✅ Review Input Workspace
- ✅ Sentiment Intelligence Card
- ✅ Rating Intelligence Card
- ✅ Emotion Intelligence Card
- ✅ Quality Assessment
- ✅ Aspect Intelligence Panel
- ✅ Evidence Viewer
- ✅ Guest Summary
- ✅ Management Summary
- ✅ Review-Level Executive Summary
- ✅ API Health Monitoring
- ✅ Keyboard Shortcuts
- ✅ Responsive Landscape Layout
- ✅ Dark Theme
- ✅ Light Theme
- ✅ Local Review History Integration

---

## History Page ✅ Complete

Route:

```text
/history
```

Capabilities:

- ✅ Analysis History
- ✅ Review Preview
- ✅ Sentiment Snapshot
- ✅ Rating Snapshot
- ✅ Timestamp Tracking
- ✅ Clear History
- ✅ Local Storage Persistence
- ✅ Click Historical Record to Reopen
- ✅ Analyze Page Input Rehydration
- ✅ Full Analysis Result Restoration
- ✅ Historical Restoration Without Re-running the API
- ✅ Refresh-Safe Selected Analysis Restoration

---

## Batch Analysis Page ✅ Complete

Route:

```text
/batch
```

Capabilities:

- ✅ CSV Upload
- ✅ Drag-and-Drop File Selection
- ✅ Papa Parse Integration
- ✅ CSV Validation
- ✅ Required `review_text` Validation
- ✅ Review-Length Validation
- ✅ Empty / Trailing Row Handling
- ✅ Validation Summary
- ✅ Review Preview
- ✅ Batch API Integration
- ✅ Loading State
- ✅ Error State
- ✅ Full-Width Results Workspace
- ✅ Review Text Results
- ✅ Sentiment Results
- ✅ Rating Results
- ✅ Emotion Results
- ✅ Primary Aspect Results
- ✅ Quality Results
- ✅ Pagination
- ✅ CSV Results Export
- ✅ Latest Successful Batch Persistence
- ✅ Responsive Horizontal Workflow

The Version 1 batch workflow is:

```text
CSV Upload
    │
    ▼
Validation
    │
    ▼
Preview
    │
    ▼
Analyze Batch
    │
    ▼
Results Table
    │
    ├── Inspect Results
    ├── Download CSV
    │
    ▼
Persist Latest Batch
```

---

# Insights Dashboard ✅ Complete

Route:

```text
/insights
```

The dashboard uses the most recently completed Batch Analysis stored in the browser.

## Portfolio KPIs

- ✅ Total Reviews
- ✅ Average Rating
- ✅ Positive Percentage
- ✅ Neutral Percentage
- ✅ Negative Percentage
- ✅ Top Emotion
- ✅ Top Positive Aspect
- ✅ Top Negative Aspect

## Sentiment Analytics

- ✅ Positive / Neutral / Negative Distribution
- ✅ Interactive Donut Visualization
- ✅ Counts and Percentages

## Rating Analytics

- ✅ Average Predicted Rating
- ✅ 1–5 Rating Distribution
- ✅ Interactive Bar Visualization

## Emotion Analytics

- ✅ Emotion Distribution
- ✅ Leading Emotion
- ✅ Interactive Horizontal Bar Visualization

## Aspect Intelligence

- ✅ Positive Aspect Counts
- ✅ Negative Aspect Counts
- ✅ Positive vs Negative Comparison
- ✅ Top Aspect Ranking
- ✅ Interactive Visualization

## Executive Summary

- ✅ Overall Portfolio Position
- ✅ Management-Level Narrative
- ✅ Leading Strength
- ✅ Improvement Opportunity
- ✅ Recommended Focus

The portfolio Executive Summary is deterministically derived from the latest batch analytics and does not require an additional model or LLM call.

---

# Frontend Analytics Architecture

Shared portfolio calculations are centralized in:

```text
frontend/src/utils/batchAnalytics.js
```

The analytics layer derives:

```text
Total Reviews

Sentiment Counts
Sentiment Percentages

Average Rating
Rating Distribution

Emotion Distribution
Top Emotion

Aspect Mentions
Positive Aspect Counts
Negative Aspect Counts

Top Positive Aspect
Top Negative Aspect
```

This shared analytics layer feeds the Insights KPIs, visualizations, and Executive Summary.

---

# Frontend Persistence

Version 1 separates individual-review history from batch analytics.

## Individual Review History

Storage:

```text
hri-history
```

Used by:

```text
Analyze
History
```

This preserves previously analyzed reviews and supports History → Analyze rehydration.

## Selected Historical Analysis

The selected historical record is additionally maintained for refresh-safe restoration while reopening a previous analysis.

## Latest Batch Analysis

Storage:

```text
hri-latest-batch
```

Used by:

```text
Batch
Insights
```

This allows the latest completed batch and Insights dashboard to survive navigation and browser refresh.

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

# Batch API Contract

## Request

```json
{
  "reviews": [
    {
      "review_text": "Great hotel with excellent service."
    },
    {
      "review_text": "The room was noisy and poorly maintained."
    }
  ]
}
```

## Response

The batch endpoint returns:

```json
{
  "results": [
    {
      "review_text": "...",
      "sentiment": {},
      "rating": {},
      "emotion": {},
      "aspects": [],
      "summary": {},
      "quality": {},
      "errors": []
    }
  ]
}
```

Each result follows the existing single-review intelligence contract.

---

# Review Intelligence Output

Each review produces:

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

## Backend Test Dashboard

Run:

```bash
python tests/run_tests.py
```

The backend test dashboard covers the major backend intelligence suites, including:

```text
API
Graph
ABSA
Emotion
Summary
```

## Execute Individual Suites

```bash
python tests/run_tests.py api

python tests/run_tests.py graph

python tests/run_tests.py absa

python tests/run_tests.py summary

python tests/run_tests.py emotion
```

## Execute Multiple Suites

```bash
python tests/run_tests.py graph summary
```

## Re-run Failed Tests

```bash
python tests/run_tests.py failed
```

---

# Logging

Logs are written under:

```text
logs/
```

Example:

```text
hotel_review_workflow_YYYY-MM-DD.log
```

Log information includes:

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
|---|---|
| Data Preparation | ✅ Complete |
| Sentiment Classification | ✅ Complete |
| Rating Prediction | ✅ Complete |
| Emotion Detection | ✅ Complete |
| ABSA | ✅ Complete |
| Reconciliation | ✅ Complete |
| Summarization | ✅ Complete |
| FastAPI API | ✅ Complete |
| Batch Processing API | ✅ Complete |
| Backend Automated Testing | ✅ Complete |
| Backend Test Dashboard | ✅ Complete |
| Sentiment Benchmark Research | ✅ Complete |
| Production Sentiment Model Selection | ✅ Complete |
| React Frontend Foundation | ✅ Complete |
| Analyze Page | ✅ Complete |
| History Page | ✅ Complete |
| Historical Analysis Rehydration | ✅ Complete |
| Theme Support | ✅ Complete |
| Batch Analysis UI | ✅ Complete |
| Batch CSV Workflow | ✅ Complete |
| Batch CSV Download | ✅ Complete |
| Insights Dashboard UI | ✅ Complete |
| Sentiment Visualization | ✅ Complete |
| Rating Visualization | ✅ Complete |
| Emotion Visualization | ✅ Complete |
| Aspect Intelligence | ✅ Complete |
| Portfolio Executive Summary | ✅ Complete |
| Frontend Production Build Validation | ✅ Complete |
| Native XLSX Export | ⏳ Deferred |
| Frontend Automated Test Suite | ⏳ Deferred |
| Production Dependency Hardening | ⏳ Deferred |
| Sentiment Gold Dataset | ⏳ Future Enhancement |
| Mixed Sentiment Framework | ⏳ Future Enhancement |
| Local LLM Support | ⏳ Future Enhancement |

---

# Version 1 Completion

## Backend Version 1 ✅ Complete

Version 1 backend development includes:

```text
Sentiment Classification
Rating Prediction
Emotion Detection
ABSA
Reconciliation
Summarization
Quality Assessment
FastAPI APIs
Batch API
LangGraph Orchestration
Logging
Automated Backend Testing
```

## Frontend Version 1 ✅ Complete

Version 1 frontend development includes:

```text
Analyze Workspace
History & Rehydration
Batch Analysis
CSV Validation
Batch Results
CSV Export
Portfolio Insights
Sentiment Visualization
Rating Visualization
Emotion Visualization
Aspect Intelligence
Executive Summary
Responsive Layout
Dark / Light Themes
```

The current Version 1 feature set is frozen unless its scope is explicitly reopened.

---

# Deferred Engineering Work

The current Version 1 is considered **development-complete**, not production-hardened.

## Frontend Testing

A dedicated frontend automated test suite using technologies such as Vitest and React Testing Library is deferred.

The existing backend automated testing remains unchanged.

## Dependency Hardening

Current frontend dependency auditing identifies dependency upgrades that require breaking major-version migrations.

These migrations are intentionally deferred rather than applying forced dependency upgrades to the stable development build.

A future production-hardening cycle should review:

```text
Vite / esbuild

React Router

Related transitive dependencies
```

## Bundle Optimization

The frontend production build completes successfully.

Advanced bundle optimization is deferred, including:

```text
Route-Level Lazy Loading

Dynamic Imports

Manual Chunking

Dependency-Level Code Splitting
```

---

# Version 1.1 / Version 2 Candidates

The following features are outside Version 1.

## Batch Enhancements

```text
Native XLSX Export

Historical Batch Library

Batch-to-Batch Comparison

Retry / Resume Workflows

Advanced Batch Filtering
```

## Insights Enhancements

```text
Cross-Batch Trend Analytics

Time-Based Sentiment Trends

Advanced Dashboard Filtering

Property / Hotel Comparison

Historical Executive Analytics

Additional Visualization Types
```

## Intelligence Enhancements

```text
Human Gold Dataset

Mixed Sentiment Framework

SetFit Evaluation

Domain-Adapted Transformers

Traveler Persona Detection

Local LLM Summaries

Advanced Operational Recommendations

RAG-Based Review Intelligence
```

## Reporting

```text
One-Click Executive Reports

Downloadable Dashboard Reports

Historical Portfolio Reports
```

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
Automated Backend Testing
React Analyze Screen
History Screen
History Rehydration
Theme Support
```

## Phase 2 ✅ Complete

```text
Batch Analysis Page
CSV Upload Workflow
CSV Validation
Batch Processing Experience
Batch Results Workspace
Download Results
Latest Batch Persistence
```

## Phase 3 ✅ Complete

```text
Insights Dashboard
Portfolio KPI Analytics
Sentiment Visualization
Rating Analytics
Emotion Analytics
Aspect Intelligence
Executive Summary
```

## Phase 4 ⏳ Future Enhancements

```text
Production Hardening
Frontend Automated Testing
Dependency Upgrades
Bundle Optimization

Historical Batch Analytics
Cross-Batch Trends
Multi-Hotel Benchmarking
Trend Forecasting
Recommendation Engine
Persona Detection
Executive Reporting
Local LLM Integration
RAG-Based Review Intelligence
```

---

# Design Principles

The platform follows a modular LangGraph architecture and a reusable component-driven frontend architecture.

Core principles:

```text
Simplicity

Traceability

Reliability

Explainability

Maintainability

Modularity

Local Execution

Minimal Dependencies

Responsive User Experience
```

The Version 1 implementation deliberately prioritizes explainable and maintainable intelligence over unnecessary model or user-interface complexity.

---

# Version 1 Final State

**Hotel Review Intelligence Version 1 is development-complete.**

The solution now provides a single local workflow for:

```text
Individual Review Intelligence

Historical Review Analysis

Batch Review Processing

Portfolio Analytics

Interactive Insights

Management-Level Executive Summary
```

Future changes should be classified as one of:

1. Defect correction
2. Production hardening
3. Version 1.1 enhancement
4. Version 2 feature development

New functionality should not be added to Version 1 unless the Version 1 scope is explicitly reopened.