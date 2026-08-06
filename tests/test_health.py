def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"

    assert data["model_loaded"] is True

    assert data["model_name"] == "resnet18_finetune"

    assert data["device"] in ["cpu", "cuda", "mps"]


def test_health_schema(client):

    response = client.get("/health")

    data = response.json()

    required_fields = {
        "status",
        "model_loaded",
        "model_name",
        "checkpoint_path",
        "device",
    }

    assert required_fields.issubset(data.keys())
