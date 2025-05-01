from transformers import pipeline

def summarize_review(review_text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # For longer reviews, split and summarize parts
    if len(review_text.split()) > 500:
        # Split into chunks and summarize each
        # Then combine summaries
        pass
    else:
        summary = summarizer(review_text, max_length=60, min_length=20, do_sample=False)
        return summary[0]['summary_text']