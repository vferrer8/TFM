"""
Script de entrenamiento XGBoost — TFM F1-GPT.

Replica el pipeline del notebook F1_XGBoost_TFM.ipynb usando los
hiperparámetros óptimos obtenidos por RandomizedSearchCV (60 iter × 5 folds).
Dataset: data/f1_dataset_v2.csv  (25 121 filas, 1950-2024)
Salida : models/race_pos_model.pkl
"""

import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import MODEL_SAVE_PATH

DATASET_PATH = "data/f1_dataset_v2.csv"
RANDOM_STATE = 42

FEATURES = [
    "grid",
    "driver_pos_before",
    "driver_wins_before",
    "constructor_pos_before",
    "driver_experience",
    "driver_age",
    "avg_finish_last3",
    "circuit_alt",
    "year",
    "season_progress",
]
TARGET = "positionOrder"

# Hiperparámetros óptimos del RandomizedSearchCV (F1_XGBoost_TFM.ipynb)
BEST_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.005,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "min_child_weight": 20,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "gamma": 2.0,
}


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.array(y_true) != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train() -> None:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset no encontrado: {DATASET_PATH}\n"
            "Asegúrate de que f1_dataset_v2.csv está en la carpeta data/."
        )

    print(f"Cargando dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"  {df.shape[0]:,} filas × {df.shape[1]} columnas  |  NaN: {df.isna().sum().sum()}")

    X = df[FEATURES]
    y = df[TARGET]

    # Partición 70% train / 15% val / 15% test (igual que el notebook)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    print("\nEntrenando XGBoost con early stopping (máx. 2000 árboles)...")
    model = xgb.XGBRegressor(
        n_estimators=2000,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **BEST_PARAMS,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    best_n = model.best_iteration
    print(f"  Mejor iteración (early stop): {best_n}")

    # Métricas en test
    y_pred_test = model.predict(X_test)
    y_pred_train_val = model.predict(X_train_val)

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae_test = float(mean_absolute_error(y_test, y_pred_test))
    mape_test = _mape(np.array(y_test), y_pred_test)
    gap = float(r2_score(y_train_val, y_pred_train_val)) - r2_test

    print(f"\n{'Métrica':<12} | {'Test':>8} | {'Referencia notebook':>20}")
    print(f"{'-'*12}-+-{'-'*8}-+-{'-'*20}")
    print(f"{'R²':<12} | {r2_test:>8.4f} | {'0.2457':>20}")
    print(f"{'RMSE (pos)':<12} | {rmse_test:>8.4f} | {'5.9482':>20}")
    print(f"{'MAE (pos)':<12} | {mae_test:>8.4f} | {'4.9143':>20}")
    print(f"{'MAPE (%)':<12} | {mape_test:>8.2f} | {'84.41':>20}")
    print(f"{'Gap ovfit.':<12} | {gap:>8.4f} | {'0.0454':>20}")

    # Guardar modelo (sin LabelEncoders — features 100% numéricas)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    save_data = {
        "model": model,
        "features": FEATURES,
        "best_iteration": best_n,
        "metrics": {
            "r2": r2_test,
            "rmse": rmse_test,
            "mae": mae_test,
            "mape": mape_test,
            "gap": gap,
        },
    }
    model_path = os.path.join(MODEL_SAVE_PATH, "race_pos_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nModelo guardado: {model_path}")


if __name__ == "__main__":
    train()
