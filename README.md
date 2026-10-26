# Hotel Review Intelligence 2.0

A modular hotel review analytics platform built with **LangGraph**, **FastAPI**, **scikit-learn**, local **Transformer models**, and a modern **React + Vite Intelligence Workspace**.

Hotel Review Intelligence 2.0 converts unstructured guest feedback into explainable sentiment, predicted rating, emotion, aspect-level intelligence, quality signals, summaries, batch analytics, and management-oriented portfolio insights.

## Version 2.0 Status

**Development Status: Complete**

Version 2.0 expands the original model-focused hotel review analyzer into an end-to-end intelligence platform covering:

- Individual hotel review analysis
- Explainable aspect evidence and quality checks
- Historical analysis restoration
- CSV batch analysis and validation
- Downloadable batch results
- Portfolio-level KPIs and interactive visualizations
- Deterministic executive insights
- Responsive light and dark themes

The backend and planned frontend feature scope for Version 2.0 are complete. Production hardening, frontend automated testing, dependency major-version upgrades, and advanced cross-batch analytics remain future work.

---

# What's New in 2.0

The earlier version proved the core ML/NLP capabilities. Version 2.0 focuses on how those capabilities work together as a usable product.

| Area | Earlier Version | Version 2.0 |
|---|---|---|
| Architecture | Model-centric analysis flow | **FastAPI + LangGraph** modular orchestration |
| Model execution | Independent analytical outputs | Validation → ML/NLP nodes → reconciliation → summarization → aggregation |
| Sentiment | Sentiment prediction | Benchmark-driven **TF-IDF + Logistic Regression** production pipeline |
| Explainability | Prediction-focused outputs | Aspect evidence, confidence, quality warnings, summaries, and operational focus |
| Frontend | Basic single-review analyzer | **React + Vite** intelligence workspace |
| History | Limited analysis flow | Previous analyses can be reopened with input and output restored |
| Batch | Not a complete workflow | CSV upload, validation, preview, processing, results, and download |
| Insights | Basic charts / isolated outputs | Portfolio KPIs plus sentiment, rating, emotion, and aspect visualizations |
| Management view | Review summary | Deterministic portfolio executive summary and recommended focus |

The main design shift is:

```text
Models → Orchestration → API → Explainability → Analytics → Business Insight
```

---

# Features

## Core Intelligence Capabilities

- Sentiment Classification
- Rating Prediction (1–5 Stars)
- Emotion Detection
- Aspect-Based Sentiment Analysis (ABSA)
- Cross-Model Reconciliation
- Rule-Based Business Summarization
- Quality Assessment
- LangGraph Workflow Orchestration
- FastAPI REST APIs
- Single Review Analysis
- Batch Review Processing
- Automated Backend Test Dashboard

## Frontend Capabilities

- React Intelligence Workspace
- Analyze Workspace
- Historical Analysis and Result Rehydration
- CSV Batch Analysis
- Batch Validation and Preview
- Batch Results Table
- Downloadable CSV Results
- Portfolio Insights Dashboard
- Sentiment Visualization
- Rating Visualization
- Emotion Visualization
- Aspect Intelligence Visualization
- Executive Summary
- Responsive Landscape-First Layout
- Dark / Light Theme Support
- Browser-Based Persistence

---

# Application Preview

Version 2.0 uses a responsive, landscape-first frontend organized around four connected workflows: **Analyze, Batch, Insights, and History**.

## Analyze Workspace

The Analyze workspace combines the review input with sentiment, predicted rating, dominant emotion, quality assessment, aspect intelligence, summaries, evidence, and operational focus in one explainable workflow.

### Ready state

![Analyze workspace ready state](docs/analyze-tab-part1.png)

### Intelligence results

![Analyze workspace intelligence results](docs/analyze-tab-part2.png)

The interface keeps the original review visible while the intelligence workspace exposes confidence-aware metrics, aspect-level sentiment, management summaries, improvement opportunities, and supporting evidence.

## Batch Analysis

Batch Analysis extends the same intelligence workflow to CSV data.

```text
Upload → Validate → Preview → Analyze → Inspect → Download
```

### Before processing

![Batch analysis before processing](docs/batch-tab-before-response.png)

### After processing

![Batch analysis results](docs/batch-tab-after-response.png)

The batch workspace validates the uploaded CSV, reports usable rows, processes the review set through the existing batch API, presents full-width structured results, and supports CSV export.

## Portfolio Insights

The Insights dashboard converts the latest completed batch into management-oriented portfolio analytics.

### KPI overview

![Portfolio insights KPI overview](docs/insights-tab-part1.png)

### Visual intelligence

![Portfolio insights visual analytics](docs/insights-tab-part2.png)

The dashboard includes sentiment distribution, rating distribution, emotion distribution, positive-vs-negative aspect intelligence, portfolio KPIs, and an executive summary derived from the same shared analytics layer.

## History and Rehydration

History stores recent individual analyses in the browser. Selecting a record restores the original review and its prior intelligence output in the Analyze workspace without rerunning inference.

![Analysis history](docs/hostory-tab.png)

## Light and Dark Themes

The workspace supports both dark and light themes and a collapsible sidebar for more working space.

![Dark mode with expanded navigation](docs/home-screen-dark-mode-with-sidebar-opened.png)

![Light mode with collapsed navigation](docs/home-screen-white-mode-with-sidebar-compressed.png)

---

# Version 2.0 User Experience

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
React + Vite Frontend
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
Visualization / Persistence / Portfolio Analytics
```

---

# Project Structure

```text
multi-ml-modelling/

├── backend/
│   ├── backend/
│   │   ├── api/
│   │   ├── nodes/
│   │   ├── subgraphs/
│   │   ├── config.py
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── logger.py
│   │   └── model_registry.py
│   ├── models/
│   ├── training/
│   ├── notebooks/
│   └── tests/
│       ├── run_tests.py
│       ├── test_api.py
│       ├── test_graph.py
│       ├── test_aspects.py
│       ├── test_emotion_graph.py
│       └── test_summarization.py
│
├── frontend/
│   ├── public/
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
│   │   ├── config/
│   │   ├── hooks/
│   │   ├── pages/
│   │   ├── styles/
│   │   ├── utils/
│   │   │   ├── batchAnalytics.js
│   │   │   └── ...
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── .env.example
│
├── docs/
│   ├── PROJECT_STATUS.md
│   ├── ARCHITECTURE.md
│   ├── MODEL_RESEARCH.md
│   ├── FRONTEND_ROADMAP.md
│   ├── analyze-tab-part1.png
│   ├── analyze-tab-part2.png
│   ├── batch-tab-after-response.png
│   ├── batch-tab-before-response.png
│   ├── home-screen-dark-mode-with-sidebar-opened.png
│   ├── home-screen-white-mode-with-sidebar-compressed.png
│   ├── hostory-tab.png
│   ├── insights-tab-part1.png
│   └── insights-tab-part2.png
│
├── data/
├── logs/
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

The Batch Analysis frontend accepts CSV input with a `review_text` column.

```csv
review_text
"Great location and friendly staff"
"Dirty room and very noisy"
"Excellent service, clean room and comfortable bed"
```

Supported workflow:

- File selection and drag-and-drop upload
- Quoted and multiline CSV parsing with Papa Parse
- Empty-row handling
- Review-length validation
- Preview before analysis
- Batch processing
- Structured result inspection
- CSV result download

The current backend batch contract supports up to **100 reviews per request**.

---

# Environment Variables

Create `.env` and configure values as required.

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

```bash
cd backend
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

Example artifacts:

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

Example artifacts:

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

A dedicated benchmarking exercise was completed before selecting the production sentiment model.

## Models Evaluated

- CountVectorizer + Logistic Regression
- TF-IDF + Logistic Regression
- TF-IDF + LinearSVC
- all-MiniLM-L6-v2 embeddings + calibrated Logistic Regression

## Benchmark Results

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| CountVectorizer + Logistic Regression | 0.8458 | 0.6842 | 0.6853 |
| TF-IDF + Logistic Regression | 0.8407 | 0.7226 | 0.7059 |
| TF-IDF + LinearSVC | 0.8707 | 0.6668 | 0.6747 |

## Production Decision

**TF-IDF + Logistic Regression** remains the selected sentiment model because it provides the strongest overall balance for the Version 2.0 requirements, including neutral-review handling and maintainability.

Sentiment-model experimentation is frozen for this release.

Deferred research includes:

- Human Gold Dataset
- Mixed Sentiment Framework
- SetFit Evaluation
- Domain-Adapted Transformers

---

# Run FastAPI

```bash
cd backend
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

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Example frontend environment:

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

Application:

```text
http://localhost:5173
```

## Frontend Build Validation

The current frontend successfully compiles with:

```bash
npm run build
```

Production hardening, dependency major-version migrations, and advanced bundle optimization remain outside the development-complete Version 2.0 scope.

---

# Frontend Modules

## Analyze

Route: `/analyze`

The Analyze workspace provides:

- Review input and sample reviews
- Sentiment intelligence
- Predicted rating
- Dominant emotion
- Quality assessment and warnings
- Aspect intelligence
- Evidence and operational focus
- Guest, management, and executive summaries
- Keyboard shortcuts
- History integration

## History

Route: `/history`

History provides:

- Recent analysis records
- Review preview
- Sentiment and rating snapshots
- Timestamp tracking
- Local persistence
- Clear history
- Click-to-reopen behavior
- Analyze workspace rehydration without re-running inference

## Batch

Route: `/batch`

Batch provides:

- CSV upload
- Validation summary
- Preview
- Batch API integration
- Full-width result inspection
- Sentiment, rating, emotion, primary aspect, and quality output
- Pagination
- CSV result export
- Latest successful batch persistence

## Insights

Route: `/insights`

Insights provides:

- Total Reviews
- Average Rating
- Positive / Neutral / Negative percentages
- Top Emotion
- Top Positive Aspect
- Top Negative Aspect
- Sentiment distribution
- Rating distribution
- Emotion distribution
- Positive vs negative aspect comparison
- Executive Summary
- Leading Strength
- Improvement Opportunity
- Recommended Focus

---

# Frontend Analytics Architecture

Shared portfolio calculations are centralized in:

```text
frontend/src/utils/batchAnalytics.js
```

The shared layer derives sentiment counts and percentages, average rating, rating distribution, emotion distribution, aspect counts, leading emotion, and leading positive/negative aspects.

The same analytics source feeds KPI cards, charts, and the deterministic Executive Summary, avoiding duplicate calculations across frontend views.

---

# Frontend Persistence

## Individual Review History

Storage key:

```text
hri-history
```

Used by Analyze and History.

## Latest Batch Analysis

Storage key:

```text
hri-latest-batch
```

Used by Batch and Insights.

This separation prevents individual-review history from interfering with portfolio analytics.

---

# API Endpoints

```http
GET /api/v1/health
GET /api/v1/models
POST /api/v1/analyze
POST /api/v1/analyze-batch
```

## Batch Request

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

## Batch Response Shape

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

---

# Review Intelligence Output

Each analyzed review produces:

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

---

# Testing

## Backend Test Dashboard

```bash
python tests/run_tests.py
```

Major backend suites include:

```text
API
Graph
ABSA
Emotion
Summary
```

Individual suites can be executed with:

```bash
python tests/run_tests.py api
python tests/run_tests.py graph
python tests/run_tests.py absa
python tests/run_tests.py summary
python tests/run_tests.py emotion
```

Re-run failed tests:

```bash
python tests/run_tests.py failed
```

Frontend production-build validation has also been completed with `npm run build`. A dedicated frontend automated test suite remains deferred.

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

Log information includes timestamp, file, module, line number, log level, and message.

---

# Current Project Status

| Component | Status |
|---|---|
| Data Preparation | Complete |
| Sentiment Classification | Complete |
| Rating Prediction | Complete |
| Emotion Detection | Complete |
| ABSA | Complete |
| Reconciliation | Complete |
| Summarization | Complete |
| FastAPI API | Complete |
| LangGraph Workflow | Complete |
| Batch Processing API | Complete |
| Backend Automated Testing | Complete |
| Sentiment Benchmark Research | Complete |
| Production Sentiment Model Selection | Complete |
| React Frontend Foundation | Complete |
| Analyze Workspace | Complete |
| History + Rehydration | Complete |
| Batch Analysis UI | Complete |
| Batch CSV Workflow | Complete |
| CSV Results Download | Complete |
| Portfolio Insights | Complete |
| Sentiment Visualization | Complete |
| Rating Visualization | Complete |
| Emotion Visualization | Complete |
| Aspect Intelligence | Complete |
| Portfolio Executive Summary | Complete |
| Dark / Light Theme | Complete |
| Frontend Build Validation | Complete |
| Native XLSX Export | Deferred |
| Frontend Automated Test Suite | Deferred |
| Production Dependency Hardening | Deferred |

---

# Version 2.0 Completion

## Backend

Version 2.0 backend capabilities include:

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

## Frontend

Version 2.0 frontend capabilities include:

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

The Version 2.0 development scope is frozen unless explicitly reopened.

---

# Deferred Engineering Work

Version 2.0 is **development-complete**, not production-hardened.

Deferred engineering work includes:

- Frontend automated testing with tools such as Vitest / React Testing Library
- Production dependency remediation
- Vite / esbuild major-version migration review
- React Router major-version migration review
- Route-level lazy loading
- Dynamic imports
- Manual chunking
- Bundle optimization

---

# Future Candidates

## Batch Enhancements

- Native XLSX export
- Historical batch library
- Batch-to-batch comparison
- Retry / resume workflows
- Advanced batch filtering

## Insights Enhancements

- Cross-batch trend analytics
- Time-based sentiment trends
- Advanced dashboard filtering
- Property / hotel comparison
- Historical executive analytics
- Additional visualization types

## Intelligence Enhancements

- Human Gold Dataset
- Mixed Sentiment Framework
- SetFit Evaluation
- Domain-Adapted Transformers
- Traveler Persona Detection
- Local LLM Summaries
- Advanced Operational Recommendations
- RAG-Based Review Intelligence

## Reporting

- One-click executive reports
- Downloadable dashboard reports
- Historical portfolio reports

---

# Design Principles

Hotel Review Intelligence 2.0 follows a modular backend and reusable component-driven frontend architecture.

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

Version 2.0 deliberately prioritizes connected, explainable, maintainable intelligence over unnecessary model or UI complexity.

---

# Version 2.0 Final State

**Hotel Review Intelligence 2.0 is development-complete.**

The project has evolved from a model-centric hotel review analyzer into a modular intelligence platform supporting:

```text
Individual Review Intelligence
Historical Review Analysis
Batch Review Processing
Portfolio Analytics
Interactive Insights
Management-Level Executive Summary
```

Future changes should be classified as:

1. Defect correction
2. Production hardening
3. Version 2.1 enhancement
4. Version 3 feature development

New functionality should not be added to Version 2.0 unless its scope is explicitly reopened.
