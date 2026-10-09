from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import json


# ==================================================
# Configuration
# ==================================================

MODEL_PATH = "model/best_purchase_model.pkl"
METADATA_PATH = "model/metadata.json"


# ==================================================
# Load Model and Metadata
# ==================================================

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

THRESHOLD = metadata["threshold"]


# ==================================================
# FastAPI Application
# ==================================================

app = FastAPI(
    title="E-Commerce Purchase Prediction API",
    description="ML API for predicting whether an online shopping session will result in a purchase.",
    version="1.0.0"
)


# ==================================================
# Serve Frontend
# ==================================================

app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)


# ==================================================
# Request Schema
# ==================================================

class UserSession(BaseModel):

    Administrative: float
    Administrative_Duration: float

    Informational: float
    Informational_Duration: float

    ProductRelated: float
    ProductRelated_Duration: float

    BounceRates: float
    ExitRates: float

    PageValues: float
    SpecialDay: float

    Month: str

    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int

    VisitorType: str

    Weekend: bool


# ==================================================
# Root Endpoint - Frontend
# ==================================================

@app.get("/")
def home():

    return FileResponse(
        "frontend/index.html"
    )


# ==================================================
# Health Check
# ==================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ==================================================
# Model Information
# ==================================================

@app.get("/model-info")
def model_info():

    return {
        "model": metadata["model"],
        "threshold": metadata["threshold"],
        "roc_auc": metadata["roc_auc"],
        "precision": metadata["precision"],
        "recall": metadata["recall"],
        "f1": metadata["f1"]
    }


# ==================================================
# Prediction Endpoint
# ==================================================

@app.post("/predict")
def predict(data: UserSession):

    df = pd.DataFrame([
        data.model_dump()
    ])

    probability = model.predict_proba(
        df
    )[0][1]

    will_purchase = (
        probability >= THRESHOLD
    )

    return {
        "purchase_probability": round(
            float(probability),
            4
        ),
        "threshold": THRESHOLD,
        "will_purchase": bool(
            will_purchase
        )
    }