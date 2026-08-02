from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException

from api.middleware import register_exception_handlers

from inference.predictor import BrainTumorPredictor

from api.schemas import (
    PredictionResponse,
    HealthResponse
)

from configs.settings import (
    API_VERSION,
    MODEL_NAME,
    MODEL_ARCHITECTURE,
)



@asynccontextmanager
async def lifespan(app: FastAPI):

    predictor = BrainTumorPredictor()

    app.state.predictor = predictor

    yield



app = FastAPI(
    title="Brain Tumor MRI API",
    version="1.0.0",
    lifespan=lifespan
)


register_exception_handlers(app)



@app.get("/")
def root():

    return {
        "project": "Brain Tumor MRI Classification",
        "model": MODEL_NAME,
        "version": API_VERSION
    }



@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    predictor = app.state.predictor

    return HealthResponse(

        status="ok",

        model_loaded=predictor.model_loaded,

        model_name=MODEL_NAME,

        architecture=MODEL_ARCHITECTURE,

        checkpoint_path=str(
            predictor.checkpoint_path.name
        ),

        device=str(
            predictor.device
        ),

        api_version=API_VERSION
    )



@app.get("/model-info")
def model_info():

    return {

        "architecture": "ResNet18",

        "training":
            "transfer learning + fine tuning",

        "classes":
        [
            "glioma_tumor",
            "meningioma_tumor",
            "no_tumor",
            "pituitary_tumor"
        ]
    }



@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    file: UploadFile = File(...)
):

    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image type"
        )


    predictor = app.state.predictor


    return predictor.predict(
        await file.read(),
        file.filename
    )
