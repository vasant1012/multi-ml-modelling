from models.model_registry import (
    get_rating_model,
    get_sentiment_model,
)


def test_models_load():

    sentiment = get_sentiment_model()

    rating = get_rating_model()

    assert sentiment is not None

    assert rating is not None