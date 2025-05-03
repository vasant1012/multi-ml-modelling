import torch


def extract_aspects(review_text, model, tokenizer, device):
    # Predefined aspect categories
    aspects = ["cleanliness", "service", "location", "amenities", "value"]

    # Process through model
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], task="aspect")

    aspect_scores = outputs["aspect"]

    # Process outputs to identify aspect sentiments
    aspect_sentiments = {}
    for i, aspect in enumerate(aspects):
        # Get sentiment score for this aspect (-1 to 1 range)
        # Classify sentiment
        sentiment_score = aspect_scores[0][0][i].mean()
        if sentiment_score > 0.3:
            sentiment = "positive"
        elif sentiment_score < -0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        aspect_sentiments[aspect] = {
            "score": sentiment_score,
            "sentiment": sentiment
        }

    return aspect_sentiments