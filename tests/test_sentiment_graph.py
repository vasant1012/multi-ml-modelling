from app.subgraphs.sentiment_graph import build_graph

def test_positive_fallback():
    out = build_graph().invoke({"cleaned_text": "great clean friendly hotel", "requested_components": ["sentiment"]})
    assert out["sentiment"]["label"] == "positive"
