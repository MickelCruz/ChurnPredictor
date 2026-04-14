from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / 'src' / 'api'))
from main import app

client = TestClient(app)

# Payload válido
customer_valido = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "mensaje" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["modelo"] == "XGBoost"
    assert "threshold" in response.json()


def test_predict_valido():
    response = client.post("/predict", json=customer_valido)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "churn_label" in data
    assert 0 <= data["churn_probability"] <= 1
    assert data["churn_prediction"] in [0, 1]
    assert data["churn_label"] in ["Churn", "No Churn"]


def test_predict_campo_faltante():
    payload_incompleto = customer_valido.copy()
    del payload_incompleto["tenure"]
    response = client.post("/predict", json=payload_incompleto)
    assert response.status_code == 422


def test_predict_tipo_incorrecto():
    payload_invalido = customer_valido.copy()
    payload_invalido["tenure"] = "doce"
    response = client.post("/predict", json=payload_invalido)
    assert response.status_code == 422


def test_predict_batch():
    response = client.post("/predict/batch", json=[customer_valido, customer_valido])
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert "churn_count" in data
    assert "churn_rate" in data
    assert len(data["predictions"]) == 2


def test_predict_batch_vacio():
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0