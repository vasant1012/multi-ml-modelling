import torch.nn as nn
from transformers import AutoModel, AutoConfig

class HotelReviewMultiTaskModel(nn.Module):
    def __init__(self, model_name="roberta-base", num_labels=5, num_emotions=6):
        super(HotelReviewMultiTaskModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Shared layers
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # Task-specific heads
        # 1. Rating prediction (regression)
        self.rating_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # 2. Sentiment classification (3 classes)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)
        )

        # 3. Emotion detection (multi-label)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotions),
            nn.Sigmoid()
        )

        # 4. Aspect extraction and classification
        self.aspect_extractor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 5)  # Assuming 5 main aspects: cleanliness, service, location, amenities, value
        )

        # 5. Text summarization (decoder setup)
        # For summarization, we'd typically use an encoder-decoder architecture
        # This is a simplified version - in practice, you might use a full seq2seq model
        self.summary_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask, task=None):
        # Get embeddings from the encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]  # [CLS] token for classification tasks
        pooled_output = self.dropout(pooled_output)

        # Return task-specific outputs based on the task parameter
        results = {}

        if task in ["all", "rating"]:
            # Rating prediction (regression)
            rating_output = self.rating_classifier(pooled_output)
            results["rating"] = rating_output

        if task in ["all", "sentiment"]:
            # Sentiment classification
            sentiment_output = self.sentiment_classifier(pooled_output)
            results["sentiment"] = sentiment_output

        if task in ["all", "emotion"]:
            # Emotion detection
            emotion_output = self.emotion_classifier(pooled_output)
            results["emotion"] = emotion_output

        if task in ["all", "aspect"]:
            # Aspect-based sentiment
            aspect_output = self.aspect_extractor(sequence_output)
            results["aspect"] = aspect_output

        if task in ["all", "summary"]:
            # For summarization, we'll need a more complex decoder setup
            # This is a placeholder for the summarization task
            summary_features = self.summary_projection(sequence_output)
            results["summary_features"] = summary_features

        return results