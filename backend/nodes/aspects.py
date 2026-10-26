"""
Aspect-Based Sentiment Analysis (ABSA)

Hotel-focused implementation.

Features:
- Hotel aspect extraction
- Aspect-level sentiment
- Evidence extraction
- Aspect prioritization
- Confidence calculation
"""

from __future__ import annotations

import re
import sys
from typing import Any

sys.path.append(
    "C:\\Users\\vasant.rajadhvax\\LocalBin_vasant.rajadhyax\\Projects\\multi-ml-modelling"
)
from backend.state import ReviewState

# ==========================================================
# Hotel Aspect Taxonomy (V2)
# ==========================================================

ASPECTS = {

    "room": [
        "room",
        "suite",
        "accommodation",
    ],

    "bed_comfort": [
        "bed",
        "mattress",
        "pillow",
        "blanket",
    ],

    "bathroom": [
        "bathroom",
        "toilet",
        "shower",
        "sink",
        "drain",
    ],

    "sleep_quality": [
        "sleep",
        "slept",
        "rest",
        "awake",
        "insomnia",
    ],

    "staff_service": [
        "staff",
        "service",
        "housekeeping",
        "employee",
        "reception",
        "front desk",
    ],

    "checkin_checkout": [
        "check in",
        "check-in",
        "checkin",
        "checkout",
        "reservation",
        "queue",
        "waited",
        "waiting",
    ],

    "management_resolution": [
        "manager",
        "management",
        "refund",
        "compensation",
        "resolved",
        "resolution",
    ],

    "cleanliness": [
        "clean",
        "dirty",
        "dust",
        "dusty",
        "stain",
        "smell",
        "odor",
        "odour",
    ],

    "maintenance": [
        "broken",
        "repair",
        "damaged",
        "clogged",
        "maintenance",
        "air conditioning",
        "ac unit",
        "air conditioner",
    ],

    "amenities": [
        "pool",
        "gym",
        "spa",
    ],

    "wifi_internet": [
        "wifi",
        "wi-fi",
        "internet",
        "network",
        "connection",
    ],

    "restaurant": [
        "restaurant",
        "dinner",
        "lunch",
        "food",
        "menu",
        "tasting menu",
    ],

    "breakfast": [
        "breakfast",
        "buffet",
        "pastry",
    ],

    "bar_lounge": [
        "bar",
        "cocktail",
        "cocktails",
        "wine",
        "drinks",
    ],

    "location": [
        "location",
        "downtown",
        "city center",
        "city centre",
    ],

    "transport_access": [
        "airport",
        "transit",
        "train",
        "walkable",
        "transport",
    ],

    "view_scenery": [
        "view",
        "ocean",
        "sea",
        "rooftop",
        "scenery",
    ],

    "family_facilities": [
        "family",
        "kids",
        "children",
        "water park",
        "waterpark",
        "game room",
    ],

    "business_facilities": [
        "business",
        "corporate",
        "meeting",
        "meetings",
        "remote work",
        "client call",
        "conference",
    ],

    "value_for_money": [
        "price",
        "cheap",
        "cost",
        "expensive",
        "value",
    ],

    "parking": [
        "parking",
        "garage",
    ],

    "noise": [
        "noise",
        "noisy",
        "loud",
        "hallway",
        "neighbors",
        "neighbours",
        "quiet",
    ],
}

# ==========================================================
# Priority Ranking
# ==========================================================

ASPECT_PRIORITY = {

    "wifi_internet": 10,
    "cleanliness": 10,
    "sleep_quality": 10,

    "staff_service": 9,
    "checkin_checkout": 9,

    "maintenance": 8,
    "bed_comfort": 8,

    "room": 7,
    "value_for_money": 7,
    "location": 7,

    "restaurant": 6,
    "breakfast": 6,
    "bar_lounge": 6,

    "parking": 5,
    "view_scenery": 5,

    "family_facilities": 5,
    "business_facilities": 5,

    "noise": 5,
}

# ==========================================================
# Sentiment Keywords
# ==========================================================

POSITIVE_WORDS = {
    "great",
    "excellent",
    "amazing",
    "stunning",
    "beautiful",
    "clean",
    "comfortable",
    "friendly",
    "helpful",
    "professional",
    "perfect",
    "love",
    "loved",
    "pleasant",
    "kind",
    "spacious",
    "breathtaking",
    "memorable",
    "unbeatable",
}

NEGATIVE_WORDS = {
    "bad",
    "poor",
    "dirty",
    "terrible",
    "awful",
    "rude",
    "slow",
    "broken",
    "clogged",
    "noisy",
    "disappointing",
    "disappointed",
    "frustrating",
    "nightmare",
    "chaos",
    "stressful",
    "uncomfortable",
    "loud",
    "problem",
    "issue",
}

# ==========================================================
# Helpers
# ==========================================================

def split_sentences(text: str) -> list[str]:
    """
    Split text into meaningful sentences.
    """
    parts = re.split(
        r"[.!?;\n]+",
        text,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def normalize(text: str) -> str:
    return text.lower().strip()


def calculate_sentiment_score(
    sentence: str,
) -> tuple[str, float]:
    """
    Determine sentiment and confidence.
    """

    text = normalize(sentence)

    positive_hits = sum(
        1
        for word in POSITIVE_WORDS
        if word in text
    )

    negative_hits = sum(
        1
        for word in NEGATIVE_WORDS
        if word in text
    )

    score = positive_hits - negative_hits

    if score > 0:
        confidence = min(
            0.60 + (score * 0.10),
            0.95,
        )

        return (
            "positive",
            round(confidence, 2),
        )

    if score < 0:
        confidence = min(
            0.60 + (abs(score) * 0.10),
            0.95,
        )

        return (
            "negative",
            round(confidence, 2),
        )

    return (
        "neutral",
        0.50,
    )

# ==========================================================
# Main ABSA Node
# ==========================================================

def analyze_aspects(
    state: ReviewState,
) -> dict[str, Any]:
    """
    LangGraph ABSA node.

    Input:
        cleaned_text

    Output:
        aspects[]
    """

    review_text = state.get(
        "cleaned_text",
        "",
    )

    if not review_text:
        return {
            "aspects": []
        }

    sentences = split_sentences(
        review_text
    )

    detected_aspects = []

    seen_pairs = set()

    for sentence in sentences:

        sentence_norm = normalize(
            sentence
        )

        for (
            aspect,
            keywords,
        ) in ASPECTS.items():

            matched = any(
                keyword in sentence_norm
                for keyword in keywords
            )

            if not matched:
                continue

            pair = (
                aspect,
                sentence,
            )

            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)

            sentiment, confidence = (
                calculate_sentiment_score(
                    sentence
                )
            )

            detected_aspects.append(
                {
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "priority":
                        ASPECT_PRIORITY.get(
                            aspect,
                            1,
                        ),
                    "evidence": sentence,
                }
            )

    detected_aspects.sort(
        key=lambda item: (
            -item["priority"],
            -item["confidence"],
        )
    )

    return {
        "aspects": detected_aspects
    }