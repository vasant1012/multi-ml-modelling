
from unittest.mock import patch

from backend.graph import build_review_graph, review_graph

# ------------------------------------------------------------------
# Mock Emotion Model
# ------------------------------------------------------------------

def fake_emotion_model(
    text,
    truncation=True,
):
    """
    Deterministic emotion response used during tests.

    Avoids downloading/loading the real HuggingFace model.
    """

    return [
        {
            "label": "joy",
            "score": 0.85,
        },
        {
            "label": "neutral",
            "score": 0.15,
        },
    ]


# ------------------------------------------------------------------
# Sentiment Tests
# ------------------------------------------------------------------

def test_positive_review():

    result = review_graph.invoke(
        {
            "review_text":
            "Amazing hotel and great staff"
        }
    )

    assert result["sentiment_label"] == "positive"

    assert result["predicted_rating"] >= 4


def test_negative_review():

    result = review_graph.invoke(
        {
            "review_text":
            "Terrible room and rude staff"
        }
    )

    assert result["sentiment_label"] == "negative"

    assert result["predicted_rating"] <= 2


# ------------------------------------------------------------------
# Structure Tests
# ------------------------------------------------------------------

def test_review_contains_expected_fields():

    result = review_graph.invoke(
        {
            "review_text":
            "Average hotel stay"
        }
    )

    assert "sentiment_label" in result
    assert "sentiment_confidence" in result

    assert "predicted_rating" in result
    assert "rating_confidence" in result


def test_confidence_range():

    result = review_graph.invoke(
        {
            "review_text":
            "Amazing staff and hotel"
        }
    )

    assert (
        0 <= result["sentiment_confidence"] <= 1
    )

    assert (
        0 <= result["rating_confidence"] <= 1
    )


# ------------------------------------------------------------------
# Emotion Tests
# ------------------------------------------------------------------

@patch(
    "app.nodes.emotion.get_emotion_model",
    return_value=fake_emotion_model,
)
def test_graph_includes_emotion(
    mock_get_model,
):

    graph = build_review_graph()

    result = graph.invoke(
        {
            "review_text":
            "Amazing hotel and great staff"
        }
    )

    assert result["sentiment_label"] == "positive"

    assert result["predicted_rating"] in range(
        1,
        6,
    )

    assert (
        result["dominant_emotion"]
        == "joy"
    )

    assert (
        0
        <= result["emotion_confidence"]
        <= 1
    )

    assert "emotion" in result["response"]

    assert (
        result["response"]["emotion"][
            "dominant"
        ]
        == "joy"
    )


@patch(
    "app.nodes.emotion.get_emotion_model",
    return_value=fake_emotion_model,
)
def test_graph_response_structure(
    mock_get_model,
):

    graph = build_review_graph()

    result = graph.invoke(
        {
            "review_text":
            "Average hotel stay"
        }
    )

    response = result["response"]

    # Base response structure

    assert "review_text" in response

    assert "sentiment" in response

    assert "rating" in response

    assert "emotion" in response

    assert "errors" in response

    # Quality block

    assert "quality" in response

    assert (
        "status"
        in response["quality"]
    )

    assert (
        "warnings"
        in response["quality"]
    )

    assert isinstance(
        response["quality"]["warnings"],
        list,
    )

    assert "summary" in response

    assert (
        "guest_summary"
        in response["summary"]
    )

    assert (
        "management_summary"
        in response["summary"]
    )

    assert (
        "positive_highlights"
        in response["summary"]
    )

    assert (
        "improvement_opportunities"
        in response["summary"]
    )

    assert (
        "executive_summary"
        in response["summary"]
    )


# ------------------------------------------------------------------
# Reconciliation Tests
# ------------------------------------------------------------------

@patch(
    "app.nodes.emotion.get_emotion_model",
    return_value=fake_emotion_model,
)
def test_graph_contains_reconciliation(
    mock_get_model,
):

    graph = build_review_graph()

    result = graph.invoke(
        {
            "review_text":
            "Amazing hotel and great staff"
        }
    )

    # LangGraph state

    assert (
        "consistency_status"
        in result
    )

    assert "warnings" in result

    # API response

    response = result["response"]

    assert "quality" in response

    assert (
        response["quality"][
            "status"
        ]
        in {
            "consistent",
            "review_required",
        }
    )

    assert isinstance(
        response["quality"]["warnings"],
        list,
    )


# ------------------------------------------------------------------
# End-to-End Response Validation
# ------------------------------------------------------------------

@patch(
    "app.nodes.emotion.get_emotion_model",
    return_value=fake_emotion_model,
)
def test_complete_response_schema(
    mock_get_model,
):

    graph = build_review_graph()

    result = graph.invoke(
        {
            "review_text":
            (
                "The hotel was clean, "
                "staff were friendly "
                "and the location was excellent."
            )
        }
    )

    response = result["response"]

    # Sentiment structure

    assert (
        "label"
        in response["sentiment"]
    )

    assert (
        "confidence"
        in response["sentiment"]
    )

    # Rating structure

    assert (
        "prediction"
        in response["rating"]
    )

    assert (
        "confidence"
        in response["rating"]
    )

    # Emotion structure

    assert (
        "dominant"
        in response["emotion"]
    )

    assert (
        "predictions"
        in response["emotion"]
    )

    assert isinstance(
        response["emotion"][
            "predictions"
        ],
        list,
    )

    assert (
        len(
            response["emotion"][
                "predictions"
            ]
        )
        > 0
    )

def test_graph_contains_summary(
    graph,
):
    """
    Summary output should be generated
    and exposed through the final API contract.
    """

    result = graph.invoke(
        {
            "review_text":
                (
                    "Excellent location but "
                    "Wi-Fi kept disconnecting."
                )
        }
    )

    assert (
        "guest_summary"
        in result
    )

    assert (
        "management_summary"
        in result
    )

    assert (
        "executive_summary"
        in result
    )

    response = result[
        "response"
    ]

    assert (
        "summary"
        in response
    )

    summary = response[
        "summary"
    ]

    assert (
        "guest_summary"
        in summary
    )

    assert (
        "management_summary"
        in summary
    )

    assert (
        "positive_highlights"
        in summary
    )

    assert (
        "improvement_opportunities"
        in summary
    )

    assert (
        "executive_summary"
        in summary
    )