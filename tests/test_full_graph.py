from app.graph import build_review_graph

def test_full_workflow():
    out = build_review_graph().invoke({"review_id":"1", "original_text":"Great location but noisy room.", "original_rating":4})
    result = out["final_result"]
    assert result["sentiment"]
    assert result["rating_prediction"]
    assert isinstance(result["aspects"], list)

def test_component_selection():
    out = build_review_graph().invoke({"original_text":"Friendly staff", "requested_components":["sentiment"]})
    assert out["final_result"]["sentiment"]
    assert out["final_result"]["rating_prediction"] is None
