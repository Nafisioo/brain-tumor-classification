import pytest

from fastapi.testclient import TestClient

from api.main import app
from inference.predictor import BrainTumorPredictor


@pytest.fixture(scope="session")
def client():
    """
    FastAPI test client with lifespan support.
    """

    with TestClient(app) as test_client:
        yield test_client



@pytest.fixture(scope="session")
def predictor():
    """
    Real production model predictor.
    """

    return BrainTumorPredictor()



@pytest.fixture
def sample_image_path():
    """
    Real MRI image for integration testing.
    """

    return (
        "data/raw/brain_mri/test/"
        "no_tumor/"
        "bright_img_32_7522.png"
    )



@pytest.fixture
def mock_predictor(monkeypatch):

    class MockPredictor:

        def predict(
            self,
            image_bytes,
            filename=None
        ):

            return {
                "model_name": "mock_model",
                "architecture": "ResNet18",
                "filename": filename,
                "class_index": 2,
                "class_name": "no_tumor",
                "confidence": 0.99,
                "probabilities": {
                    "glioma_tumor":0.001,
                    "meningioma_tumor":0.001,
                    "no_tumor":0.997,
                    "pituitary_tumor":0.001
                }
            }


    app.state.predictor = MockPredictor()

    return app.state.predictor