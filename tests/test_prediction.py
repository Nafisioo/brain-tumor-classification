import math



def test_prediction_success(
    client,
    sample_image_path
):


    with open(
        sample_image_path,
        "rb"
    ) as image:


        response = client.post(
            "/predict",
            files={
                "file": (
                    "sample.png",
                    image,
                    "image/png"
                )
            }
        )


    assert response.status_code == 200


    result = response.json()


    assert result["model_name"] == (
        "resnet18_finetune"
    )


    assert result["class_name"] in [
        "glioma_tumor",
        "meningioma_tumor",
        "no_tumor",
        "pituitary_tumor"
    ]


    assert 0 <= result["confidence"] <= 1



def test_probability_distribution(
    client,
    sample_image_path
):


    with open(
        sample_image_path,
        "rb"
    ) as image:


        response = client.post(
            "/predict",
            files={
                "file": (
                    "sample.png",
                    image,
                    "image/png"
                )
            }
        )


    result = response.json()


    probabilities = (
        result["probabilities"]
    )


    assert len(probabilities) == 4


    total_probability = sum(
        probabilities.values()
    )


    assert math.isclose(
        total_probability,
        1.0,
        abs_tol=1e-5
    )



def test_invalid_file_upload(client):


    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"hello world",
                "text/plain"
            )
        }
    )


    assert response.status_code == 400