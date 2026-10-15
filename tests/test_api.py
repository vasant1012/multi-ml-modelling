def test_analyze_endpoint(client):

    response = client.post(
        "/api/v1/analyze",
        json={
            "review_text":
            "Amazing hotel and great staff"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "review_text" in data

    assert "sentiment" in data

    assert "rating" in data

def test_health_endpoint(client):

    response = client.get(
        "/api/v1/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }

def test_models_endpoint(client):

    response = client.get(
        "/api/v1/models"
    )

    assert response.status_code == 200

    data = response.json()

    assert "sentiment_model" in data

    assert "rating_model" in data

def test_empty_review_validation(client):

    response = client.post(
        "/api/v1/analyze",
        json={
            "review_text": ""
        }
    )

    assert response.status_code == 422


def test_analyze_response_contains_emotion(client):
    response = client.post(
        "/api/v1/analyze",
        json={
            "review_text": (
                "I loved the room and the staff were wonderful."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "emotion" in data
    assert "dominant" in data["emotion"]
    assert "confidence" in data["emotion"]
    assert "predictions" in data["emotion"]

    assert 0 <= data["emotion"]["confidence"] <= 1

def test_analyze_response_contains_quality(
    client,
):
    """The single-review API should expose quality checks."""

    response = client.post(
        "/api/v1/analyze",
        json={
            "review_text": (
                "Amazing hotel and great staff"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "quality" in data
    assert "status" in data["quality"]
    assert "warnings" in data["quality"]

    assert data["quality"]["status"] in {
        "consistent",
        "review_required",
    }

    assert isinstance(
        data["quality"]["warnings"],
        list,
    )