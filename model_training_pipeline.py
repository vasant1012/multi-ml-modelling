from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torch.nn as nn

class HotelReviewDataset(Dataset):
    def __init__(self, reviews, ratings, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.reviews = reviews
        self.ratings = ratings
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        rating = self.ratings[idx]

        encoding = self.tokenizer(
            review,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'rating': torch.tensor(rating, dtype=torch.float)
        }

def train_model(model, train_dataloader, val_dataloader, device, epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ratings = batch['rating'].to(device)

            optimizer.zero_grad()

            # Multi-task forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, task="all")

            # Calculate losses for each task
            rating_loss = nn.MSELoss()(outputs['rating'].squeeze(), ratings)

            # More loss calculations would be added for other tasks
            # sentiment_loss = ...
            # emotion_loss = ...

            # Combined loss (with weights if needed)
            total_loss = rating_loss  # + sentiment_loss + emotion_loss + ...

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()

        # Validation step
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        # Add validation metrics here