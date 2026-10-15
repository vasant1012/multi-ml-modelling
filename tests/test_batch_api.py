def test_batch_endpoint(client):

    response = client.post(
        "/api/v1/analyze-batch",
        json={
            "reviews": [
                {
                    "review_text":
                    "Amazing hotel"
                },
                {
                    "review_text":
                    "Dirty room"
                }
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "results" in data

    assert len(data["results"]) == 2

def test_batch_result_structure(client):

    response = client.post(
        "/api/v1/analyze-batch",
        json={
            "reviews": [
                {
                    "review_text":
                    "Good location"
                }
            ]
        },
    )

    result = response.json()[
        "results"
    ][0]

    assert "review_text" in result

    assert "sentiment" in result

    assert "rating" in result

def test_large_batch(client):

    batch = []

    for i in range(25):
        batch.append(
            {
                "review_text":
                f"Test review {i}"
            }
        )

    response = client.post(
        "/api/v1/analyze-batch",
        json={
            "reviews": batch
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["results"]) == 25

def test_batch_results_contain_emotions(client):
    response = client.post(
        "/api/v1/analyze-batch",
        json={
            "reviews": [
                {
                    "review_text": (
                        "Wonderful hotel and helpful staff."
                    )
                },
                {
                    "review_text": (
                        "The room was dirty and disappointing."
                    )
                },
            ]
        },
    )

    assert response.status_code == 200

    results = response.json()["results"]

    assert len(results) == 2

    for result in results:
        assert "emotion" in result
        assert "dominant" in result["emotion"]
        assert "predictions" in result["emotion"]

def test_batch_results_contain_quality(
    client,
):
    """Every batch item should contain reconciliation output."""

    response = client.post(
        "/api/v1/analyze-batch",
        json={
            "reviews": [
                {
                    "review_text": (
                        "Amazing location and helpful staff."
                    )
                },
                {
                    "review_text": (
                        "Dirty room and terrible service."
                    )
                },
            ]
        },
    )

    assert response.status_code == 200

    results = response.json()["results"]

    for result in results:
        assert "quality" in result
        assert result["quality"]["status"] in {
            "consistent",
            "review_required",
        }
        assert isinstance(
            result["quality"]["warnings"],
            list,
        )