from app.subgraphs.rating_graph import build_graph

def test_rating_is_in_range():
    out = build_graph().invoke({"cleaned_text": "terrible dirty room", "requested_components": ["rating"]})
    assert 1 <= out["rating_prediction"]["rating"] <= 5
