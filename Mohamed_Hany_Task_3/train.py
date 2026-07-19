"""
train.py
========
Trains the Uber Fare Prediction pipeline from scratch using feature_engineering.py.
All features are recomputed — no pre-computed columns from the CSV are used.
This guarantees zero Training/Serving Skew with the Flask app.

Usage
-----
    python train.py          # standard training
    python train.py --tune   # + RandomizedSearchCV
    python train.py --gpu    # GPU training
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from feature_engineering import FEATURE_COLUMNS, create_features

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("data/final_internship_data.csv")
MODEL_OUT    = Path("models/catboost_fare_pipeline.joblib")
TARGET       = "fare_amount"
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# NYC bounding box in radians
_LON_MIN = np.deg2rad(-74.30); _LON_MAX = np.deg2rad(-73.65)
_LAT_MIN = np.deg2rad(40.45);  _LAT_MAX = np.deg2rad(40.95)

NUMERIC_FEATS     = [c for c in FEATURE_COLUMNS if c != "Weather"]
CATEGORICAL_FEATS = ["Weather"]

# ── Load & clean ──────────────────────────────────────────────────────────────

def load_and_clean(path: Path) -> pd.DataFrame:
    print(f"[data] Loading {path} ...")
    df = pd.read_csv(path)
    n0 = len(df)

    # Drop PII / identifiers
    df.drop(columns=["User ID","User Name","Driver Name","key"], errors="ignore", inplace=True)

    # Drop pre-computed columns — we recompute from coordinates
    df.drop(
        columns=["distance","bearing","jfk_dist","lga_dist","nyc_dist",
                 "ewr_dist","sol_dist","hour","day","weekday","month","year"],
        errors="ignore", inplace=True,
    )

    # Drop rows with missing coords / target
    df.dropna(subset=["pickup_longitude","pickup_latitude",
                       "dropoff_longitude","dropoff_latitude",
                       "passenger_count","fare_amount","pickup_datetime",
                       "Car Condition","Traffic Condition","Weather"], inplace=True)

    # Fare bounds
    df = df[(df[TARGET] >= 2.50) & (df[TARGET] <= 150.0)]

    # NYC bounding box
    df = df[
        df["pickup_longitude"].between(_LON_MIN, _LON_MAX)
        & df["pickup_latitude"].between(_LAT_MIN, _LAT_MAX)
        & df["dropoff_longitude"].between(_LON_MIN, _LON_MAX)
        & df["dropoff_latitude"].between(_LAT_MIN, _LAT_MAX)
    ]

    # Passenger count
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 4)]

    n1 = len(df)
    print(f"[data] Rows after cleaning: {n1:,} ({100*n1/n0:.1f}% retained)")
     
    return df

    print(df["pickup_datetime"].min())
    print(df["pickup_datetime"].max())

# ── Feature engineering ───────────────────────────────────────────────────────

def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print("[fe]   Running create_features() ...")
    y = df[TARGET].copy()
    X = create_features(df)

    # Filter out zero-distance trips
    mask = X["distance"] > 0.0
    X, y = X[mask], y[mask]
    print(f"[fe]   Shape: {X.shape}")
    return X, y


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_pipeline(use_gpu: bool = False) -> Pipeline:
    pre = ColumnTransformer([
        ("num",     RobustScaler(),                              NUMERIC_FEATS),
        ("weather", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False),         CATEGORICAL_FEATS),
    ])
    cb_params: dict = {
        "loss_function":  "RMSE",
        "iterations":     500,
        "depth":          8,
        "learning_rate":  0.1,
        "l2_leaf_reg":    5,
        "subsample":      0.8,
        "bootstrap_type": "Bernoulli",
        "random_state":   RANDOM_STATE,
        "verbose":        100,
    }
    if use_gpu:
        cb_params.update({"task_type": "GPU", "devices": "0"})
    return Pipeline([("preprocessor", pre), ("model", CatBoostRegressor(**cb_params))])


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, label="Test"):
    pred = pipeline.predict(X)
    mae  = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2   = r2_score(y, pred)
    print(f"\n[{label}]  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
    return mae, rmse, r2


# ── Main ──────────────────────────────────────────────────────────────────────

def train(tune: bool = False, use_gpu: bool = False) -> None:
    t0 = time.time()

    df = load_and_clean(DATA_PATH)
    X, y = prepare(df)
    tmp = X.copy()
    tmp["fare"] = y.values

    print(tmp[["distance", "fare"]].describe())

    print(tmp.sort_values("distance")[["distance", "fare"]].head(20))

    print(tmp[tmp["distance"] < 1][["distance", "fare"]].describe())

    print("Rows with distance < 1 km:", len(tmp[tmp["distance"] < 1]))
    print(
    tmp[tmp["distance"] < 1]
         .sort_values("fare", ascending=False)
         .head(30))  

    X_tr, X_te, y_tr, y_te = train_test_split(


        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE

    )
    print(f"[split] Train={len(X_tr):,}  Test={len(X_te):,}")

    pipeline = build_pipeline(use_gpu=use_gpu)

    if tune:
        print("[tune]  RandomizedSearchCV n_iter=20 cv=3 ...")
        search = RandomizedSearchCV(
            pipeline,
            param_distributions={
                "model__iterations":    [300, 500, 700],
                "model__depth":         [6, 7, 8],
                "model__learning_rate": [0.05, 0.1, 0.15],
                "model__l2_leaf_reg":   [3, 5, 7],
            },
            n_iter=20, cv=3,
            scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        print(f"[tune]  Best params: {search.best_params_}")
    else:
        print("[train] Fitting pipeline ...")
        pipeline.fit(X_tr, y_tr)
        pre = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        feature_names = pre.get_feature_names_out()

        import pandas as pd

        imp = pd.DataFrame({
         "Feature": feature_names,
            "Importance": model.feature_importances_
              }).sort_values("Importance", ascending=False)

        print(imp)
        best = pipeline

    evaluate(best, X_tr, y_tr, "Train")
    evaluate(best, X_te, y_te, "Test ")

    joblib.dump(best, MODEL_OUT)
    print(f"\n[done]  Saved -> {MODEL_OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--gpu",  action="store_true")
    args = ap.parse_args()
    train(tune=args.tune, use_gpu=args.gpu)
