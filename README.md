# ChurnPredictor — Sistema de Predicción de Churn

![CI](https://github.com/MickelCruz/ChurnPredictor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Model](https://img.shields.io/badge/model-XGBoost-orange)
![API](https://img.shields.io/badge/api-FastAPI-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)

Sistema end-to-end de predicción de churn para telecomunicaciones. Detecta clientes en riesgo de abandonar el servicio usando XGBoost con un pipeline MLOps completo.

[English version](README_EN.md)

---

## Problema de Negocio

El 26.6% de los clientes abandona el servicio. Con 7,032 clientes en el dataset, eso representa 1,869 churners — un volumen significativo que justifica 
la inversión en un sistema de detección temprana.

**Impacto estimado del modelo:** Con Recall de 0.80 y threshold optimizado de 0.566, el modelo detecta 4 de cada 5 churners, reduciendo intervenciones innecesarias 
en un 50% comparado con intervenir a todos los clientes sin modelo.

---

## Hallazgos del EDA

| Variable | Hallazgo |
|---|---|
| `Contract` | Clientes mes a mes tienen 42.7% de churn vs 2.8% en contratos de 2 años |
| `PaymentMethod` | Cheque electrónico tiene 45.3% de churn vs ~15% en métodos automáticos |
| `InternetService` | Fibra óptica tiene 41.9% de churn — posiblemente por alto costo |
| `tenure` | Correlación -0.35 con churn — los primeros meses son el período crítico |
| `SeniorCitizen` | Clientes mayores tienen 41.7% de churn vs 23.7% |
| `gender` | Diferencia mínima (27% vs 26.2%) — no es predictor relevante |

**Perfil de mayor riesgo:** cliente con contrato mes a mes, tenure bajo, fibra óptica, sin seguridad online y pago con cheque electrónico.

---

## Resultados del Modelado

Se evaluaron 4 modelos con Optuna (50 trials + CV 5 folds):

| Modelo | AUC-PR Test | GAP | Recall |
|---|---|---|---|
| Logistic Regression | 0.5842 | 0.0891 | 0.7679 |
| Random Forest | 0.5972 | 0.1350 | 0.7571 |
| **XGBoost** | **0.6159** | **0.0938** | **0.7964** |
| LightGBM | 0.6114 | 0.1200 | 0.7964 |

**XGBoost seleccionado** — mejor AUC-PR Test y GAP más controlado entre modelos avanzados.

**Nota sobre el GAP:** La learning curve confirmó que el GAP de 0.09 es una limitación del tamaño del dataset (7,032 registros), no overfitting. Con más datos las curvas convergerían.

---

## Features Más Importantes (SHAP)

| Ranking | Feature | Dirección |
|---|---|---|
| 1 | `Contract_Month-to-month` | ↑ churn |
| 2 | `tenure` | ↓ churn a mayor antigüedad |
| 3 | `InternetService_Fiber optic` | ↑ churn |
| 4 | `OnlineSecurity_No` | ↑ churn |
| 5 | `TechSupport_No` | ↑ churn |
| 6 | `MonthlyCharges` | ↑ churn a mayor cargo |
| 7 | `PaymentMethod_Electronic check` | ↑ churn |
| 8 | `Contract_Two year` | ↓ churn |

---

## Stack Técnico

| Categoría | Tecnologías |
|---|---|
| Modelo | XGBoost, LightGBM, scikit-learn, Optuna, SHAP |
| API | FastAPI, Pydantic V2, Uvicorn |
| Tests | pytest, httpx |
| Monitoring | Evidently |
| Orquestación | Prefect 3 |
| Infraestructura | Docker, GitHub Actions |
| Datos | Parquet (pyarrow) |

---

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Estado de la API |
| GET | `/health` | Health check |
| POST | `/predict` | Predicción individual |
| POST | `/predict/batch` | Predicción por lote |

---

## Decisiones Técnicas

- **Threshold 0.566** — optimizado para maximizar F1, no se usa 0.5 por defecto.
- **AUC-PR como métrica principal** — superior a ROC-AUC para datasets desbalanceados.
- **Cross-validation en Optuna** — 5 folds estratificados para evitar sobreajuste.
- **Parquet sobre CSV** — preserva tipos de datos y más eficiente.

---

## Autor

**Mickel Cruz** — ML/MLOps Engineer

[![GitHub](https://img.shields.io/badge/GitHub-MickelCruz-black)](https://github.com/MickelCruz)