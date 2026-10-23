# Architecture

## High-Level System

```text
Frontend (React)
        │
        ▼

FastAPI Backend

        │
        ▼

LangGraph Workflow

        │
        ▼

Sentiment
Rating
Emotion
ABSA

        │
        ▼

Reconciliation

        │
        ▼

Summarization

        │
        ▼

Response Aggregation
```

---

# Backend Workflow

```text
Validation
     │

Preprocessing
     │

 ┌──────────────┬──────────────┬──────────────┐
 │              │              │              │
 ▼              ▼              ▼              ▼

Sentiment     Rating        Emotion         ABSA

 └──────────────┴──────────────┴──────────────┘
                     │
                     ▼

              Reconciliation

                     │
                     ▼

               Summarization

                     │
                     ▼

                Aggregation
```

---

# Backend Technology Stack

```text
Python

FastAPI

LangGraph

Scikit-Learn

Transformers

PyTorch

Pandas

Pytest
```

---

# Frontend Technology Stack

```text
React

Vite

Axios

React Router

Lucide Icons

Responsive CSS

Local Storage
```

---

# Data Flow

```text
Review
  ↓

Analyze API

  ↓

LangGraph

  ↓

Sentiment

Rating

Emotion

ABSA

  ↓

Summary

  ↓

Frontend Visualization
```

---

# Design Principles

- Explainability First
- Local Execution
- Modularity
- Minimal Dependencies
- Maintainability
- Domain-Focused Analytics