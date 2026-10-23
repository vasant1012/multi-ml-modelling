# Model Research

## Objective

Identify the best sentiment-classification approach for the hotel review dataset.

---

# Dataset

Labels were derived from ratings.

```text
1-2 Stars  -> Negative

3 Stars    -> Neutral

4-5 Stars  -> Positive
```

Train:

```text
14,343
```

Validation:

```text
2,049
```

Test:

```text
4,099
```

---

# Models Evaluated

## Experiment 1

CountVectorizer + Logistic Regression

Results:

```text
Accuracy            0.8458
Balanced Accuracy   0.6842
Macro F1            0.6853
```

Verdict:

Baseline Alternative

---

## Experiment 2

TF-IDF + Logistic Regression

Results:

```text
Accuracy            0.8407
Balanced Accuracy   0.7226
Macro F1            0.7059
```

Verdict:

✅ Selected Production Model

---

## Experiment 3

TF-IDF + LinearSVC

Results:

```text
Accuracy            0.8707
Balanced Accuracy   0.6668
Macro F1            0.6747
```

Verdict:

Higher Accuracy but weaker minority-class performance.

---

## Experiment 4

MiniLM Embeddings

Model:

```text
all-MiniLM-L6-v2
```

Results:

```text
Accuracy            0.8409
Balanced Accuracy   0.6160
Macro F1            0.6108
```

Findings:

- Better calibration
- Worse Neutral Recall
- Worse Macro F1

Verdict:

Rejected for V1

---

# Final Decision

Production Sentiment Model:

```text
TF-IDF + Logistic Regression
```

---

# Future Research

Deferred

```text
Gold Dataset

Mixed Sentiment

SetFit

Domain Adaptation

Transformer Fine-Tuning
```