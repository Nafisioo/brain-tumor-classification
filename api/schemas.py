from typing import Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_name: str
    checkpoint_path: str
    device: str


class PredictionResponse(BaseModel):
    model_name: str = Field(..., description="Model used for inference.")
    filename: Optional[str] = Field(
        default=None,
        description="Original uploaded filename."
    )
    class_index: int = Field(..., ge=0)
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]