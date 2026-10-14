# Hotel Review Intelligence

A deliberately simple end-to-end reference project that uses:

- scikit-learn for supervised sentiment and 1-to-5 rating models
- lightweight deterministic logic for emotion, aspect sentiment and summarization
- LangGraph for workflow orchestration
- FastAPI for the backend API
- React + Vite for the user interface

The deterministic components keep the project easy to run without paid APIs. They are intentionally replaceable with transformer or LLM implementations later.

## 1. Expected CSV

Place the input file at `data/raw/hotel_reviews.csv` with two columns:

```csv
review,rating
"Great location and friendly staff",5
"Dirty room and very noisy",1
```

Column names can be changed through environment variables `TEXT_COLUMN` and `RATING_COLUMN`.

## 2. Backend setup

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
python -m training.prepare_data
python -m training.train_sentiment
python -m training.train_rating
python -m training.evaluate
uvicorn app.api.main:app --reload
```

Open API documentation at `http://localhost:8000/docs`.

## 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## 4. Tests

```bash
pytest -q
```

Tests work without trained artifacts because inference nodes include clearly marked rule-based fallbacks. Train the models before using real data.

## 5. Main workflow

`validate -> preprocess -> five component subgraphs in parallel -> reconcile -> aggregate`

The five component results are:

1. positive, neutral or negative sentiment
2. predicted rating from 1 through 5
3. one or more emotions
4. sentiment for hotel aspects with evidence
5. a concise extractive summary

## 6. Important simplifications

- Emotion detection uses transparent keyword scores.
- ABSA uses a controlled aspect vocabulary and nearby sentiment words.
- Summarization is extractive, so it does not invent facts.
- Sentiment and rating fall back to rules if model files do not exist.

These choices are appropriate for a working baseline. Replace one node at a time after establishing evaluation data.
