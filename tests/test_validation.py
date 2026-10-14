from app.nodes.validation import validate_input

def test_empty_review_is_rejected():
    assert validate_input({"original_text": ""})["errors"]

def test_valid_review_has_no_errors():
    assert validate_input({"original_text": "Clean room", "original_rating": 5})["errors"] == []
