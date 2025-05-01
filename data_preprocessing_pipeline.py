import re
# import nltk
# from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

def preprocess_text(text):
    # Basic cleaning
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()

    # Remove stopwords (optional - modern transformers handle these well)
    # stop_words = set(stopwords.words('english'))
    # words = text.split()
    # text = ' '.join([word for word in words if word not in stop_words])

    return text

def prepare_dataset(df):
    # Clean text
    df['clean_review'] = df['review'].apply(preprocess_text)

    # Create sentiment labels based on ratings
    df['sentiment'] = df['rating'].apply(lambda x:
                                        'negative' if x <= 2 else
                                        'neutral' if x == 3 else
                                        'positive')

    # Train-val-test split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    return train_df, val_df, test_df