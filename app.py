from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="E-Commerce Purchase Prediction API")

model = joblib.load("model/purchase_prediction.pkl")


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


@app.get("/")
def home():
    return {"message": "E-Commerce Purchase Prediction API is running"}


@app.post("/predict")
def predict(data: UserSession):
    df = pd.DataFrame([data.model_dump()])

    probability = model.predict_proba(df)[0][1]

    return {
        "purchase_probability": round(float(probability), 4),
        "will_purchase": bool(probability >= 0.5)
    }