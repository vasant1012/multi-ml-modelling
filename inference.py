import torch
from data_preprocessing_pipeline import preprocess_text
from aspect_based_sentiment import extract_aspects
from summarizer import summarize_review


def process_review(review_text, multi_task_model, summarizer, tokenizer, device):
    # 1. Preprocess
    cleaned_text = preprocess_text(review_text)

    # 2. Run through multi-task model
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = multi_task_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            task="all"
        )

    # 3. Extract results
    predicted_rating = outputs["rating"].item()

    sentiment_probs = torch.softmax(outputs["sentiment"][0], dim=0)
    sentiment_id = torch.argmax(sentiment_probs).item()
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_sentiment = sentiment_map[sentiment_id]

    emotion_probabilities = outputs["emotion"][0]
    emotions = ["joy", "anger", "sadness", "surprise", "fear", "disgust"]
    detected_emotions = [
        emotions[i] for i in range(len(emotions))
        if emotion_probabilities[i] > 0.5
    ]

    # 4. Get aspect-based sentiments
    aspect_sentiments = extract_aspects(cleaned_text, multi_task_model, tokenizer, device)

    # 5. Generate summary
    summary = summarize_review(review_text)

    # 6. Return comprehensive analysis
    return {
        "rating": predicted_rating,
        "sentiment": predicted_sentiment,
        "emotions": detected_emotions,
        "aspects": aspect_sentiments,
        "summary": summary
    }