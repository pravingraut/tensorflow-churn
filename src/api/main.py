# /api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
from tensorflow.keras.models import load_model

app = FastAPI(title="Churn Prediction API")

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '../../artifacts')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/best_model.keras')

# Load preprocessor and model at startup
preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, 'preprocessor.joblib'))
model = load_model(MODEL_PATH)

# Define expected JSON structure (only include a subset or all features)
class Customer(BaseModel):
    # Example — adjust to include all required fields
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

@app.post("/predict")
def predict(customer: Customer):
    # Convert to DataFrame for transformer
    import pandas as pd
    data = pd.DataFrame([customer.model_dump()])
    X = preprocessor.transform(data)
    prob = model.predict(X)[0][0]
    return {"churn_probability": float(prob), "will_churn": bool(prob > 0.5)}