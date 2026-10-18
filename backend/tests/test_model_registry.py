from backend.models.model_registry import (
    get_emotion_model,
    get_rating_model,
    get_sentiment_model,
)


def test_models_load():

    sentiment = get_sentiment_model()

    rating = get_rating_model()

    assert sentiment is not None

    assert rating is not None


def test_emotion_model_falls_back_when_hf_is_unavailable(monkeypatch):
    def raise_offline_error(*args, **kwargs):
        raise OSError("No internet")

    monkeypatch.setattr(
        "backend.models.model_registry.pipeline",
        raise_offline_error,
    )

    model = get_emotion_model()
    result = model("I loved the room and staff were amazing.")

    assert callable(model)
    assert isinstance(result, list)
    assert result[0]["label"] in {"joy", "neutral", "sadness", "anger", "fear", "surprise"}


def test_stale_pickle_retrains_when_multi_class_is_missing(monkeypatch):
    get_sentiment_model.cache_clear()

    class FakeStaleModel:
        def predict_proba(self, _texts):
            raise AttributeError("'LogisticRegression' object has no attribute 'multi_class'")

    monkeypatch.setattr(
        "backend.models.model_registry.joblib.load",
        lambda path: FakeStaleModel(),
    )

    called = {}

    def fake_train():
        called["used"] = True
        return object()

    monkeypatch.setattr(
        "backend.models.model_registry._train_and_save_sentiment_model",
        fake_train,
    )

    model = get_sentiment_model()

    assert called["used"] is True
    assert model is not None