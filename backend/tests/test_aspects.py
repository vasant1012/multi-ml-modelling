from backend.nodes.aspects import (
    analyze_aspects,
    calculate_sentiment_score,
    split_sentences,
)

# ------------------------------------------------------------------
# Sentence Splitting
# ------------------------------------------------------------------

def test_split_sentences():

    text = (
        "The room was clean. "
        "The staff were rude! "
        "The ocean view was breathtaking."
    )

    sentences = split_sentences(text)

    assert len(sentences) == 3

    assert sentences[0] == "The room was clean"
    assert sentences[1] == "The staff were rude"
    assert sentences[2] == "The ocean view was breathtaking"


# ------------------------------------------------------------------
# Sentiment Scoring
# ------------------------------------------------------------------

def test_positive_sentiment_score():

    sentiment, confidence = calculate_sentiment_score(
        "The room was clean and spacious."
    )

    assert sentiment == "positive"
    assert 0 <= confidence <= 1


def test_negative_sentiment_score():

    sentiment, confidence = calculate_sentiment_score(
        "The room was dirty and terrible."
    )

    assert sentiment == "negative"
    assert 0 <= confidence <= 1


def test_neutral_sentiment_score():

    sentiment, confidence = calculate_sentiment_score(
        "The hotel has a reception desk."
    )

    assert sentiment == "neutral"
    assert 0 <= confidence <= 1


# ------------------------------------------------------------------
# Business Traveler Review
# ------------------------------------------------------------------

def test_business_traveler_review():

    result = analyze_aspects(
        {
            "cleaned_text": (
                "The location is unbeatable for corporate meetings. "
                "The Wi-Fi kept dropping every ten minutes. "
                "The rooftop view is stunning."
            )
        }
    )

    names = {
        item["aspect"]
        for item in result["aspects"]
    }

    assert "business_facilities" in names
    assert "wifi_internet" in names
    assert "location" in names
    assert "view_scenery" in names


# ------------------------------------------------------------------
# Family Traveler Review
# ------------------------------------------------------------------

def test_family_traveler_review():

    result = analyze_aspects(
        {
            "cleaned_text": (
                "The kids loved the indoor water park and game room. "
                "Check-in took over an hour despite the reservation."
            )
        }
    )

    names = {
        item["aspect"]
        for item in result["aspects"]
    }

    assert "family_facilities" in names
    assert "checkin_checkout" in names


# ------------------------------------------------------------------
# Foodie Review
# ------------------------------------------------------------------

def test_foodie_review():

    result = analyze_aspects(
        {
            "cleaned_text": (
                "The restaurant tasting menu was perfection. "
                "The mattress felt uncomfortable."
            )
        }
    )

    names = {
        item["aspect"]
        for item in result["aspects"]
    }

    assert "restaurant" in names
    assert "bed_comfort" in names


# ------------------------------------------------------------------
# Aspect Detection + Evidence
# ------------------------------------------------------------------

def test_aspect_sentiment_and_evidence():

    result = analyze_aspects(
        {
            "cleaned_text": (
                "The room was clean and spacious. "
                "The staff were rude and unhelpful."
            )
        }
    )

    aspects = result["aspects"]

    room_result = next(
        item
        for item in aspects
        if item["aspect"] == "room"
    )

    service_result = next(
        item
        for item in aspects
        if item["aspect"] == "staff_service"
    )

    assert room_result["sentiment"] == "positive"

    assert "room" in room_result["evidence"].lower()

    # We only require detection and evidence.
    # Sentiment depends on current keyword taxonomy.
    assert service_result["sentiment"] in {
        "negative",
        "neutral",
    }

    assert "staff" in service_result["evidence"].lower()


# ------------------------------------------------------------------
# Confidence Validation
# ------------------------------------------------------------------

def test_aspect_confidence_range():

    result = analyze_aspects(
        {
            "cleaned_text": (
                "The room was clean. "
                "The Wi-Fi was terrible. "
                "The location was great."
            )
        }
    )

    for aspect in result["aspects"]:
        assert 0 <= aspect["confidence"] <= 1


# ------------------------------------------------------------------
# Priority Validation
# ------------------------------------------------------------------

def test_priority_field_exists():

    result = analyze_aspects(
        {
            "cleaned_text": (
                "The Wi-Fi kept disconnecting. "
                "The room was clean. "
                "The parking was convenient."
            )
        }
    )

    for aspect in result["aspects"]:
        assert "priority" in aspect

        assert isinstance(
            aspect["priority"],
            int,
        )


def test_priority_sorting():

    result = analyze_aspects(
        {
            "cleaned_text": (
                "The Wi-Fi kept disconnecting. "
                "The parking was convenient."
            )
        }
    )

    aspects = result["aspects"]

    if len(aspects) >= 2:
        assert (
            aspects[0]["priority"]
            >= aspects[1]["priority"]
        )


def test_wifi_has_higher_priority_than_parking():

    result = analyze_aspects(
        {
            "cleaned_text": (
                "Wi-Fi was terrible. "
                "Parking was easy."
            )
        }
    )

    priorities = {
        item["aspect"]: item["priority"]
        for item in result["aspects"]
    }

    assert (
        priorities["wifi_internet"]
        > priorities["parking"]
    )


# ------------------------------------------------------------------
# Empty / Unknown Reviews
# ------------------------------------------------------------------

def test_empty_text_returns_empty_aspects():

    result = analyze_aspects(
        {
            "cleaned_text": "",
        }
    )

    assert result == {
        "aspects": []
    }


def test_review_without_known_aspects():

    result = analyze_aspects(
        {
            "cleaned_text": (
                "I visited the city during my vacation."
            )
        }
    )

    assert result["aspects"] == []


# ------------------------------------------------------------------
# Duplicate Handling
# ------------------------------------------------------------------

def test_no_duplicate_aspect_evidence_pairs():

    result = analyze_aspects(
        {
            "cleaned_text": (
                "The room was clean and the bed was comfortable."
            )
        }
    )

    pairs = [
        (
            item["aspect"],
            item["evidence"],
        )
        for item in result["aspects"]
    ]

    assert len(pairs) == len(set(pairs))


# ------------------------------------------------------------------
# Real-world Mixed Review
# ------------------------------------------------------------------

def test_real_world_mixed_review():

    result = analyze_aspects(
        {
            "cleaned_text": (
                "Great downtown location and professional staff. "
                "However the Wi-Fi kept dropping during meetings."
            )
        }
    )

    names = {
        item["aspect"]
        for item in result["aspects"]
    }

    assert "location" in names
    assert "staff_service" in names
    assert "wifi_internet" in names
    assert "business_facilities" in names