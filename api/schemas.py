from typing import Dict, Optional, List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    API health status response.
    """

    status: str = Field(
        default="ok",
        description="API health status."
    )

    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded successfully."
    )

    model_name: str = Field(
        ...,
        description="Active model name."
    )

    architecture: str = Field(
        ...,
        description="Deep learning architecture."
    )

    checkpoint_path: str = Field(
        ...,
        description="Path of loaded model checkpoint."
    )

    device: str = Field(
        ...,
        description="Inference device used by PyTorch."
    )

    api_version: str = Field(
        ...,
        description="API version."
    )


class PredictionResponse(BaseModel):
    """
    Model prediction output.
    """

    model_name: str = Field(
        ...,
        description="Model used for inference."
    )

    architecture: str = Field(
        ...,
        description="Model architecture."
    )

    filename: Optional[str] = Field(
        default=None,
        description="Original uploaded filename."
    )

    class_index: int = Field(
        ...,
        ge=0,
        description="Predicted class index."
    )

    class_name: str = Field(
        ...,
        description="Predicted tumor category."
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score."
    )

    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across classes."
    )