from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from typing import List

from backend.predict_v3 import predict_image


# =========================================================
# Response Models
# =========================================================

class TopPrediction(BaseModel):
    class_name: str
    probability: float


class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float

    top3: List[dict]

    entropy: float
    margin: float

    feature_similarity: float

    ood: bool
    ood_threshold: float

    nearest_eurosat_class: str

    reliability_score: float
    reliability_status: str


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="AI Land Cover Classification API",
    description="""
    V3 AI land-cover classification API.

    Uses a ResNet50 model trained on EuroSAT land-cover imagery.

    The system provides:
    - Land-cover classification
    - Prediction confidence
    - Top-3 predictions
    - Prediction entropy
    - Prediction margin
    - Feature-space similarity
    - OOD (Out-of-Distribution) detection
    - Nearest EuroSAT class
    - Domain Fit / Reliability score
    """,
    version="3.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# Health Check
# =========================================================

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Check API health"
)
def health():

    return {
        "status": "ok",
        "model": "ResNet50",
        "version": "V3"
    }


# =========================================================
# Prediction Endpoint
# =========================================================

@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    summary="Classify a land-cover image",
    description="""
    Upload an image to classify it into one of the 10 EuroSAT
    land-cover classes.

    The endpoint also performs feature-space OOD detection.

    Images with feature similarity below the OOD threshold
    are flagged as Possible OOD.
    """
)
async def predict(
    file: UploadFile = File(
        ...,
        description="Image file to classify"
    )
):

    # -----------------------------------------------------
    # Check filename
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided."
        )

    # -----------------------------------------------------
    # Read uploaded file
    # -----------------------------------------------------

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # -----------------------------------------------------
    # Convert bytes → PIL image
    # -----------------------------------------------------

    try:

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image."
        )

    # -----------------------------------------------------
    # Run V3 prediction
    # -----------------------------------------------------

    try:

        result = predict_image(image)

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        )

    # -----------------------------------------------------
    # Add filename
    # -----------------------------------------------------

    result["filename"] = file.filename

    return result