from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from api.schemas import HealthResponse, PredictionResponse
from inference.predictor import BrainTumorPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = BrainTumorPredictor()
    yield


app = FastAPI(
    title="Brain Tumor MRI Classification API",
    description="Inference API for the fine-tuned ResNet18 brain tumor classifier.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "Brain Tumor MRI Classification API",
        "health": "/health",
        "docs": "/docs",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    predictor = app.state.predictor
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=predictor.model_name,
        checkpoint_path=str(predictor.checkpoint_path),
        device=str(predictor.device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file (PNG, JPG, JPEG).",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    predictor = app.state.predictor
    result = predictor.predict(image_bytes)
    result["filename"] = file.filename

    return PredictionResponse(**result)