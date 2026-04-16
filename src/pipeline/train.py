import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from prefect import flow, task

# Rutas
BASE_PATH   = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_PATH / 'models'
PARQUET_PATH = BASE_PATH.parent / 'data' / 'processed' / 'telco_churn_clean.parquet'

# Features
NUM_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CAT_FEATURES = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod']


@task(name="cargar_datos")
def cargar_datos():
    df = pd.read_parquet(PARQUET_PATH)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y


@task(name="dividir_datos")
def dividir_datos(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


@task(name="construir_pipeline")
def construir_pipeline():
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUM_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_FEATURES)
    ])
    modelo = XGBClassifier(
        n_estimators=208,
        max_depth=3,
        learning_rate=0.048,
        subsample=0.723,
        colsample_bytree=0.750,
        reg_alpha=0.045,
        reg_lambda=0.025,
        min_child_weight=10,
        scale_pos_weight=5163/1869,
        random_state=42,
        eval_metric='aucpr'
    )
    return Pipeline([('preprocessor', preprocessor), ('classifier', modelo)])


@task(name="entrenar_modelo")
def entrenar_modelo(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


@task(name="evaluar_modelo")
def evaluar_modelo(pipeline, X_test, y_test):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc_pr  = average_precision_score(y_test, y_proba)
    return round(auc_pr, 4)


@task(name="guardar_modelo")
def guardar_modelo(pipeline, auc_pr):
    joblib.dump(pipeline, DATA_PATH / 'xgb_pipeline.joblib')

    with open(DATA_PATH / 'resultados_finales.json') as f:
        config = json.load(f)

    config['metricas']['auc_pr_retrain'] = auc_pr
    with open(DATA_PATH / 'resultados_finales.json', 'w') as f:
        json.dump(config, f, indent=4)


@flow(name="churn_retraining_pipeline")
def retraining_pipeline():
    X, y                                        = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)
    pipeline                                    = construir_pipeline()
    pipeline                                    = entrenar_modelo(pipeline, X_train, y_train)
    auc_pr                                      = evaluar_modelo(pipeline, X_test, y_test)
    guardar_modelo(pipeline, auc_pr)
    print(f"Reentrenamiento completado — AUC-PR: {auc_pr}")


if __name__ == "__main__":
    retraining_pipeline()