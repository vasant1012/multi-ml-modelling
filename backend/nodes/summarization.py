"""
Rule-Based Review Summarization

This node consumes:

- sentiment
- rating
- emotion
- aspects
- reconciliation

and generates:

- guest_summary
- management_summary
- positive_highlights
- improvement_opportunities
- executive_summary
"""

from __future__ import annotations

from typing import Any

from backend.state import ReviewState

# ==========================================================
# Improvement Recommendations
# ==========================================================

IMPROVEMENT_MAP = {
    "wifi_internet":
        "Improve internet stability and Wi-Fi coverage.",

    "sleep_quality":
        "Reduce disturbances and improve guest sleep quality.",

    "noise":
        "Reduce noise from neighboring rooms and common areas.",

    "cleanliness":
        "Strengthen housekeeping and cleanliness inspections.",

    "staff_service":
        "Improve staff responsiveness and guest communication.",

    "checkin_checkout":
        "Reduce waiting times during check-in and check-out.",

    "management_resolution":
        "Improve complaint handling and service recovery.",

    "maintenance":
        "Address maintenance issues proactively.",

    "bed_comfort":
        "Improve bedding and mattress comfort.",

    "bathroom":
        "Improve bathroom maintenance and cleanliness.",

    "restaurant":
        "Review restaurant service and food consistency.",

    "breakfast":
        "Improve breakfast variety and quality.",

    "parking":
        "Improve parking accessibility and convenience.",

    "family_facilities":
        "Enhance family-focused amenities and activities.",

    "business_facilities":
        "Improve facilities for business travelers.",
}


# ==========================================================
# Helpers
# ==========================================================

def _get_positive_aspects(
    aspects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        item
        for item in aspects
        if item["sentiment"] == "positive"
    ]


def _get_negative_aspects(
    aspects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        item
        for item in aspects
        if item["sentiment"] == "negative"
    ]


def _build_guest_summary(
    positive_aspects,
    negative_aspects,
):
    """
    Customer-facing summary.
    """

    positives = [
        item["aspect"].replace("_", " ")
        for item in positive_aspects[:3]
    ]

    negatives = [
        item["aspect"].replace("_", " ")
        for item in negative_aspects[:3]
    ]

    if positives and negatives:
        return (
            f"Guests appreciated "
            f"{', '.join(positives)}, "
            f"but experienced issues with "
            f"{', '.join(negatives)}."
        )

    if positives:
        return (
            f"Guests highlighted "
            f"{', '.join(positives)} "
            f"as key strengths."
        )

    if negatives:
        return (
            f"The stay was impacted by issues with "
            f"{', '.join(negatives)}."
        )

    return (
        "The review contains a balanced mix "
        "of observations."
    )


def _build_management_summary(
    rating: int,
    sentiment: str,
    positive_aspects,
    negative_aspects,
):
    """
    Operations-focused summary.
    """

    top_positive = [
        item["aspect"].replace("_", " ")
        for item in positive_aspects[:2]
    ]

    top_negative = [
        item["aspect"].replace("_", " ")
        for item in negative_aspects[:2]
    ]

    parts = []

    parts.append(
        f"Predicted rating is {rating}/5 "
        f"with overall {sentiment} sentiment."
    )

    if top_positive:
        parts.append(
            "Guests praised "
            + ", ".join(top_positive)
            + "."
        )

    if top_negative:
        parts.append(
            "Primary improvement areas include "
            + ", ".join(top_negative)
            + "."
        )

    return " ".join(parts)


def _build_positive_highlights(
    positive_aspects,
):
    highlights = []

    for aspect in positive_aspects:

        highlights.append(
            f"Positive feedback for "
            f"{aspect['aspect'].replace('_', ' ')}"
        )

    return highlights[:5]


def _build_improvements(
    negative_aspects,
):
    opportunities = []

    for aspect in negative_aspects:

        opportunities.append(
            IMPROVEMENT_MAP.get(
                aspect["aspect"],
                (
                    "Review guest feedback "
                    "for additional improvements."
                ),
            )
        )

    return list(
        dict.fromkeys(opportunities)
    )


def _build_executive_summary(
    sentiment: str,
    rating: int,
    emotion: str,
    positive_count: int,
    negative_count: int,
):
    """
    Leadership-level summary.
    """

    if positive_count > negative_count:
        experience = "mostly positive"

    elif negative_count > positive_count:
        experience = "mostly negative"

    else:
        experience = "mixed"

    return (
        f"Guest experience appears {experience}, "
        f"with predicted rating {rating}/5, "
        f"overall sentiment {sentiment}, "
        f"and dominant emotion {emotion}."
    )


# ==========================================================
# Main Node
# ==========================================================

def generate_summary(
    state: ReviewState,
) -> dict[str, Any]:
    """
    LangGraph summarization node.
    """

    aspects = state.get(
        "aspects",
        [],
    )

    sentiment = state.get(
        "sentiment_label",
        "neutral",
    )

    rating = state.get(
        "predicted_rating",
        3,
    )

    emotion = state.get(
        "dominant_emotion",
        "neutral",
    )

    positive_aspects = _get_positive_aspects(
        aspects
    )

    negative_aspects = _get_negative_aspects(
        aspects
    )

    guest_summary = _build_guest_summary(
        positive_aspects,
        negative_aspects,
    )

    management_summary = (
        _build_management_summary(
            rating,
            sentiment,
            positive_aspects,
            negative_aspects,
        )
    )

    positive_highlights = (
        _build_positive_highlights(
            positive_aspects
        )
    )

    improvement_opportunities = (
        _build_improvements(
            negative_aspects
        )
    )

    executive_summary = (
        _build_executive_summary(
            sentiment,
            rating,
            emotion,
            len(positive_aspects),
            len(negative_aspects),
        )
    )

    return {
        "guest_summary":
            guest_summary,

        "management_summary":
            management_summary,

        "positive_highlights":
            positive_highlights,

        "improvement_opportunities":
            improvement_opportunities,

        "executive_summary":
            executive_summary,
    }