import pandas as pd
import torch
from transformers import RobertaForSequenceClassification
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data_preprocessing_pipeline import prepare_dataset
from model_training_pipeline import HotelReviewDataset
from model import HotelReviewMultiTaskModel
from aspect_based_sentiment import extract_aspects
from summarizer import summarize_review
from inference import process_review


def load_data(file_path):
    df = pd.read_csv(file_path)
    # print(df.sample(10))
    return df


def data_processing(df):
    # Example usage:
    # 1. Data Exploration and Preprocessing (already done in the provided code)
    train_df, val_df, test_df = prepare_dataset(df)
    train_dataset = HotelReviewDataset(train_df['clean_review'].tolist(), train_df['rating'].tolist(), tokenizer)
    val_dataset = HotelReviewDataset(val_df['clean_review'].tolist(), val_df['rating'].tolist(), tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Adjust batch size
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    return train_dataloader, val_dataloader

def load_pretrain_model():
    # 2. Build and train the base sentiment/rating model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")  # Example tokenizer
    model = HotelReviewMultiTaskModel().to(device)
    return device, tokenizer, model

def train_model(model, train_dataloader, val_dataloader, device, model_file_path, epochs=5):
    # 3. Add emotion detection capabilities (already integrated in the model)
    train_model(model, train_dataloader, val_dataloader, device, epochs=5)
    torch.save(model.state_dict(), f'{model_file_path}/hotel_review_model.pt')



if __name__ == '__main__':

    # Path to the locally saved model directory
    local_model_path = "hotel_review_model.pt"

    # # Load tokenizer and model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    # model = RobertaForSequenceClassification.from_pretrained(local_model_path)
    model = HotelReviewMultiTaskModel()
    model.load_state_dict(torch.load(local_model_path), map_location=torch.device('cpu'))
    model.eval()
    
    # 4. Implement the aspect-based sentiment analysis (function provided)
    # Example
    # example_review = "The hotel was very clean, but the service was slow."
    df = load_data('hotel_reviews.csv')
    example_review = df['review'].sample(n=1).values[0]
    print(example_review)
    # aspect_results = extract_aspects(example_review, model, tokenizer, device)
    # print(f"Aspect-based sentiment analysis: {aspect_results}")

    # # # 5. Integrate text summarization features (function provided)
    # # # Example
    # summary = summarize_review(example_review)
    # print(f"Summary: {summary}")


    # # 6. Create evaluation framework (not implemented in this example)
    # # This would require calculating the specified metrics on the test set.

    # # 7. Fine-tune the entire system end-to-end (already handled by train_model function)

    # Example Inference
    results = process_review(example_review, model, summarize_review, tokenizer, device)
    print(f"Complete Analysis:{results}")






