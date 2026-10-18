"""
Unit tests for backend.nodes.summarization.
"""

from backend.nodes.summarization import (
    generate_summary,
)


def test_positive_review_summary():

    state = {
        "sentiment_label": "positive",
        "predicted_rating": 5,
        "dominant_emotion": "joy",

        "aspects": [
            {
                "aspect": "location",
                "sentiment": "positive",
                "confidence": 0.90,
            },
            {
                "aspect": "staff_service",
                "sentiment": "positive",
                "confidence": 0.88,
            },
        ],
    }

    result = generate_summary(
        state
    )

    assert "guest_summary" in result

    assert "management_summary" in result

    assert (
        len(
            result["positive_highlights"]
        )
        > 0
    )

    assert (
        len(
            result[
                "improvement_opportunities"
            ]
        )
        == 0
    )


def test_negative_review_generates_improvements():

    state = {
        "sentiment_label": "negative",
        "predicted_rating": 2,
        "dominant_emotion": "anger",

        "aspects": [
            {
                "aspect": "wifi_internet",
                "sentiment": "negative",
                "confidence": 0.91,
            },
            {
                "aspect": "cleanliness",
                "sentiment": "negative",
                "confidence": 0.87,
            },
        ],
    }

    result = generate_summary(
        state
    )

    assert (
        len(
            result[
                "improvement_opportunities"
            ]
        )
        >= 2
    )

    assert (
        "executive_summary"
        in result
    )


def test_mixed_review_summary():

    state = {

        "sentiment_label":
            "neutral",

        "predicted_rating":
            3,

        "dominant_emotion":
            "neutral",

        "aspects": [

            {
                "aspect":
                    "location",

                "sentiment":
                    "positive",

                "confidence":
                    0.9,
            },

            {
                "aspect":
                    "wifi_internet",

                "sentiment":
                    "negative",

                "confidence":
                    0.9,
            },
        ],
    }

    result = generate_summary(
        state
    )

    summary = result[
        "guest_summary"
    ].lower()

    assert (
        "appreciated"
        in summary
        or
        "issues"
        in summary
    )


def test_summary_strings_exist():

    result = generate_summary(
        {
            "sentiment_label":
                "positive",

            "predicted_rating":
                4,

            "dominant_emotion":
                "joy",

            "aspects": [],
        }
    )

    assert isinstance(
        result["guest_summary"],
        str,
    )

    assert isinstance(
        result[
            "management_summary"
        ],
        str,
    )

    assert isinstance(
        result[
            "executive_summary"
        ],
        str,
    )


def test_empty_aspects():

    result = generate_summary(
        {
            "sentiment_label":
                "neutral",

            "predicted_rating":
                3,

            "dominant_emotion":
                "neutral",

            "aspects": [],
        }
    )

    assert (
        len(
            result[
                "positive_highlights"
            ]
        )
        == 0
    )

    assert (
        len(
            result[
                "improvement_opportunities"
            ]
        )
        == 0
    )
