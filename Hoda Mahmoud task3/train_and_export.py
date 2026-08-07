from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from uber_fare_pipeline import RAW_REQUIRED_COLUMNS, TripFeatureEngineer

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required to train this project.") from exc


DATASET_COLUMN_ALIASES = {
    "User ID": "User_ID",
    "User Name": "User_Name",
    "Driver Name": "Driver_Name",
    "Car Condition": "Car_Condition",
    "Traffic Condition": "Traffic_Conditions",
    "Traffic_Conditions": "Traffic_Conditions",
    "Key": "Key",
    "key": "Key",
    "Fare Amount": "Fare_Amount",
    "Fare_Amount": "Fare_Amount",
    "fare_amount": "Fare_Amount",
    "Pickup Datetime": "Pickup_Datetime",
    "pickup_datetime": "Pickup_Datetime",
    "Pickup_Datetime": "Pickup_Datetime",
    "Pickup Longitude": "Pickup_Longitude",
    "pickup_longitude": "Pickup_Longitude",
    "Pickup_Lon": "Pickup_Longitude",
    "Pickup Latitude": "Pickup_Latitude",
    "pickup_latitude": "Pickup_Latitude",
    "Dropoff Longitude": "Dropoff_Longitude",
    "dropoff_longitude": "Dropoff_Longitude",
    "Dropoff Latitude": "Dropoff_Latitude",
    "dropoff_latitude": "Dropoff_Latitude",
    "Passenger Count": "Passenger_Count",
    "passenger_count": "Passenger_Count",
    "Hour": "Hour",
    "hour": "Hour",
    "weekday": "Day",
    "Weekday": "Day",
    "Day": "Day",
    "day": "Day",
    "Month": "Month",
    "month": "Month",
    "Year": "Year",
    "year": "Year",
    "Week": "Week",
    "week": "Week",
    "JFK_Dist": "JFK_Dist",
    "jfk_dist": "JFK_Dist",
    "EWR_Dist": "EWR_Dist",
    "ewr_dist": "EWR_Dist",
    "LGA_Dist": "LGA_Dist",
    "lga_dist": "LGA_Dist",
    "SOL_Dist": "SOL_Dist",
    "sol_dist": "SOL_Dist",
    "Distance": "Distance",
    "distance": "Distance",
    "Bearing": "Bearing",
    "bearing": "Bearing",
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: DATASET_COLUMN_ALIASES.get(column, column) for column in frame.columns}
    return frame.rename(columns=renamed)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    frame = normalize_columns(frame)
    return frame


def clean_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    # Aligned with Task 2 cleaning: drop non-predictive ID/name columns when present.
    for column in ["User_ID", "User Name", "User_Name", "Driver_Name", "Driver Name"]:
        if column in frame.columns:
            frame = frame.drop(columns=column)

    # Aligned with Task 2 cleaning: deduplicate by key when a key column exists.
    if "Key" in frame.columns:
        frame = frame.drop_duplicates(subset=["Key"]).copy()
        frame = frame.drop(columns=["Key"])
    else:
        frame = frame.drop_duplicates().copy()

    if "Fare_Amount" not in frame.columns:
        raise ValueError("Target column Fare_Amount was not found in the dataset.")

    # Aligned with Task 2 cleaning: remove missing/invalid target rows and keep only non-negative fares.
    frame = frame.dropna(subset=["Fare_Amount"]).copy()
    frame = frame[frame["Fare_Amount"] >= 0].copy()

    # Aligned with Task 2 cleaning: parse datetime only to remove invalid rows, then drop it.
    if "Pickup_Datetime" in frame.columns:
        frame["Pickup_Datetime"] = pd.to_datetime(frame["Pickup_Datetime"], errors="coerce")
        frame = frame.dropna(subset=["Pickup_Datetime"]).copy()
        frame = frame.drop(columns=["Pickup_Datetime"])
    elif "pickup_datetime" in frame.columns:
        frame["pickup_datetime"] = pd.to_datetime(frame["pickup_datetime"], errors="coerce")
        frame = frame.dropna(subset=["pickup_datetime"]).copy()
        frame = frame.drop(columns=["pickup_datetime"])

    # Aligned with Task 2 cleaning: drop airport/location distance columns Task 2 did not keep.
    for column in ["JFK_Dist", "EWR_Dist", "LGA_Dist", "SOL_Dist", "NYC_Dist"]:
        if column in frame.columns:
            frame = frame.drop(columns=[column])

    # Aligned with Task 2 cleaning: drop any remaining rows with missing values in training data.
    frame = frame.dropna().copy()

    return frame


TASK2_NUMERIC_FEATURE_CANDIDATES = [
    "Passenger_Count",
    "Pickup_Longitude",
    "Pickup_Latitude",
    "Dropoff_Longitude",
    "Dropoff_Latitude",
    "Hour",
    "Day",
    "Month",
    "Year",
    "Week",
    "Distance",
    "Bearing",
]
TASK2_CATEGORICAL_FEATURE_CANDIDATES = ["Car_Condition", "Weather", "Traffic_Conditions"]


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def make_model_pipeline(model, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", TripFeatureEngineer(include_time_features=False, include_airport_features=False)),
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )


def infer_preprocessor_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_features = [column for column in TASK2_NUMERIC_FEATURE_CANDIDATES if column in frame.columns]
    categorical_features = [column for column in TASK2_CATEGORICAL_FEATURE_CANDIDATES if column in frame.columns]
    return numeric_features, categorical_features


def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = pipeline.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": r2_score(y_test, predictions),
    }


def compare_models(X_train, X_test, y_train, y_test, random_state: int = 42):
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.08,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    results = []
    fitted_pipelines: dict[str, Pipeline] = {}

    numeric_features, categorical_features = infer_preprocessor_features(X_train)

    for name, model in candidates.items():
        pipeline = make_model_pipeline(model, numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_pipeline(pipeline, X_test, y_test)
        results.append({"Model": name, **metrics})
        fitted_pipelines[name] = pipeline

    results_frame = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
    return results_frame, fitted_pipelines


def tune_xgboost(X_train, y_train, random_state: int = 42):
    search_space = {
        "model__n_estimators": [200, 300, 500, 700],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "model__reg_lambda": [1.0, 2.0, 5.0, 10.0],
    }
    numeric_features, categorical_features = infer_preprocessor_features(X_train)
    base_pipeline = make_model_pipeline(
        XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        ),
        numeric_features,
        categorical_features,
    )
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=search_space,
        n_iter=6,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def main():
    parser = argparse.ArgumentParser(description="Train and export the Uber fare model.")
    parser.add_argument("--data", type=Path, default=Path("final_internship_data.csv"), help="Path to the dataset CSV")
    parser.add_argument("--model-out", type=Path, default=Path("uber_fare_model.pkl"), help="Path to save the fitted pipeline")
    parser.add_argument("--results-out", type=Path, default=Path("model_comparison.csv"), help="Path to save comparison metrics")
    args = parser.parse_args()

    frame = load_dataset(args.data)
    frame = clean_training_frame(frame)

    X = frame.drop(columns=["Fare_Amount"])
    y = frame["Fare_Amount"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    training_cap = min(len(X_train), 50000)
    X_train_small = X_train.sample(n=training_cap, random_state=42) if len(X_train) > training_cap else X_train.copy()
    y_train_small = y_train.loc[X_train_small.index].copy()

    comparison_table, fitted_pipelines = compare_models(X_train_small, X_test, y_train_small, y_test)

    tuned_xgb = tune_xgboost(X_train_small, y_train_small)
    tuned_pipeline = tuned_xgb.best_estimator_
    tuned_metrics = evaluate_pipeline(tuned_pipeline, X_test, y_test)

    comparison_table = pd.concat(
        [
            comparison_table,
            pd.DataFrame(
                [
                    {
                        "Model": "XGBoost (Tuned)",
                        **tuned_metrics,
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values("RMSE", ascending=True).reset_index(drop=True)

    best_row = comparison_table.iloc[0]
    best_model_name = best_row["Model"]
    if best_model_name == "Linear Regression":
        final_pipeline = fitted_pipelines["Linear Regression"]
    elif best_model_name == "Random Forest":
        final_pipeline = fitted_pipelines["Random Forest"]
    elif best_model_name == "XGBoost":
        final_pipeline = fitted_pipelines["XGBoost"]
    else:
        final_pipeline = tuned_pipeline

    final_pipeline.fit(X_train, y_train)
    joblib.dump(final_pipeline, args.model_out)
    comparison_table.to_csv(args.results_out, index=False)

    print("Comparison table:")
    print(comparison_table.to_string(index=False))
    print(f"\nSaved model to: {args.model_out.resolve()}")
    print(f"Saved comparison metrics to: {args.results_out.resolve()}")


if __name__ == "__main__":
    main()