"""
Train models with TripFeatureEngineer inside the pipeline, then save the best one.
Run:  .venv/bin/python train_and_save.py
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from trip_transformer import RAW_FEATURE_COLUMNS, TripFeatureEngineer

RANDOM_STATE = 42


def load_clean_xy():
    df = pd.read_csv("final_internship_sample.csv")
    df = df.dropna()
    df = df[df["fare_amount"] > 0]
    df = df[df["passenger_count"] > 0]
    df = df[(df["distance"] > 0) & (df["distance"] < 500)]
    coord_ok = (
        (df["pickup_longitude"] < 0)
        & (df["dropoff_longitude"] < 0)
        & (df["pickup_latitude"] > 0)
        & (df["dropoff_latitude"] > 0)
    )
    df = df[coord_ok]
    y = df["fare_amount"]
    X = df[RAW_FEATURE_COLUMNS].copy()
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def make_preprocessor():
    ordinal_cols = ["Car Condition", "Traffic Condition"]
    ordinal_categories = [
        ["Bad", "Good", "Very Good", "Excellent"],
        ["Flow Traffic", "Dense Traffic", "Congested Traffic"],
    ]
    onehot_cols = ["Weather"]
    skewed_numeric = ["distance", "jfk_dist", "ewr_dist", "lga_dist", "sol_dist", "nyc_dist"]
    other_numeric = [
        "passenger_count",
        "month",
        "year",
        "bearing",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
    ]
    binary_passthrough = ["is_weekend", "is_airport_trip"]
    log1p_transformer = FunctionTransformer(np.log1p, feature_names_out="one-to-one")

    return ColumnTransformer(
        transformers=[
            (
                "ord",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encode", OrdinalEncoder(categories=ordinal_categories)),
                        ("scale", StandardScaler()),
                    ]
                ),
                ordinal_cols,
            ),
            (
                "onehot",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                onehot_cols,
            ),
            (
                "skewed",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("log", log1p_transformer),
                        ("scale", StandardScaler()),
                    ]
                ),
                skewed_numeric,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                other_numeric,
            ),
            ("bin", "passthrough", binary_passthrough),
        ]
    )


def wrap(model):
    pipe = Pipeline(
        [
            ("engineer", TripFeatureEngineer()),
            ("preprocess", make_preprocessor()),
            (
                "select",
                SelectFromModel(
                    Lasso(alpha=0.01, max_iter=20000, random_state=RANDOM_STATE),
                    threshold="median",
                ),
            ),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"{name:30s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    X_train, X_test, y_train, y_test = load_clean_xy()
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Raw columns:", list(X_train.columns))
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    lr = wrap(LinearRegression())
    lr.fit(X_train, y_train)
    rows = [evaluate("Linear Regression", lr, X_test, y_test)]

    dt = wrap(DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=12, min_samples_leaf=5))
    dt.fit(X_train, y_train)
    rows.append(evaluate("Decision Tree", dt, X_test, y_test))

    gb = wrap(GradientBoostingRegressor(random_state=RANDOM_STATE))
    gb.fit(X_train, y_train)
    rows.append(evaluate("Gradient Boosting", gb, X_test, y_test))

    rf = wrap(RandomForestRegressor(random_state=RANDOM_STATE))
    param_dist = {
        "regressor__model__n_estimators": [100, 200, 300, 400],
        "regressor__model__max_depth": [None, 5, 10, 15, 20],
        "regressor__model__min_samples_split": [2, 5, 10],
        "regressor__model__max_features": ["sqrt", "log2", 0.5, 1.0],
    }
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=15,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    rows.append(evaluate("Tuned Random Forest", best_rf, X_test, y_test))

    results = pd.DataFrame(rows).sort_values("RMSE")
    print("\nComparison table:")
    print(results.to_string(index=False))

    candidates = {
        "Linear Regression": lr,
        "Decision Tree": dt,
        "Gradient Boosting": gb,
        "Tuned Random Forest": best_rf,
    }
    winner_name = results.iloc[0]["Model"]
    winner = candidates[winner_name]
    joblib.dump(winner, "final_pipeline.joblib")
    print(f"\nSaved final_pipeline.joblib  (chosen: {winner_name})")
    print("Pipeline steps:", list(winner.regressor_.named_steps.keys()))

    sample = X_test.iloc[[0]]
    pred = winner.predict(sample)[0]
    print(f"Sanity prediction on first test row: ${pred:.2f} (actual ${y_test.iloc[0]:.2f})")


if __name__ == "__main__":
    main()
