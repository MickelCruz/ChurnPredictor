from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd

# Rutas
BASE_PATH   = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_PATH / 'models' / 'xgb_pipeline.joblib'
CONFIG_PATH = BASE_PATH / 'models' / 'resultados_finales.json'

# Cargar modelo y configuración
pipeline  = joblib.load(MODEL_PATH)
with open(CONFIG_PATH) as f:
    config = json.load(f)

THRESHOLD = config['threshold']

# Inicializar app
app = FastAPI(
    title="ChurnPredictor API",
    description="API para predecir churn de clientes de telecomunicaciones",
    version="1.0.0"
)


class CustomerFeatures(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35, "TotalCharges": 844.20
    }}}

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
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
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {
        "mensaje": "ChurnPredictor API activa",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo": config['modelo'],
        "threshold": THRESHOLD
    }


@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        # Convertir input a DataFrame
        data = pd.DataFrame([customer.model_dump()])

        # Obtener probabilidad de churn
        churn_proba = pipeline.predict_proba(data)[0][1]

        # Aplicar threshold óptimo
        churn_prediction = int(churn_proba >= THRESHOLD)

        return {
            "churn_probability": round(float(churn_proba), 4),
            "churn_prediction": churn_prediction,
            "churn_label": "Churn" if churn_prediction == 1 else "No Churn",
            "threshold_used": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerFeatures]):
    try:
        if len(customers) == 0:
            return {"total": 0, "churn_count": 0, "churn_rate": 0.0, "predictions": []}

        data = pd.DataFrame([c.model_dump() for c in customers])
        churn_probas = pipeline.predict_proba(data)[:, 1]
        churn_predictions = (churn_probas >= THRESHOLD).astype(int)

        return {
            "total": len(customers),
            "churn_count": int(churn_predictions.sum()),
            "churn_rate": round(float(churn_predictions.mean()), 4),
            "predictions": [
                {
                    "churn_probability": round(float(p), 4),
                    "churn_prediction": int(pred),
                    "churn_label": "Churn" if pred == 1 else "No Churn"
                }
                for p, pred in zip(churn_probas, churn_predictions)
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))