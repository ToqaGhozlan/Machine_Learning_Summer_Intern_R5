"""
Sklearn transformer that turns raw trip columns into model features.

Kept as an importable module so joblib can unpickle the fitted pipeline
outside the notebook (Flask). The notebook wires this in as the first
pipeline step — see the 'Trip feature engineering inside the pipeline' section.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EARTH_RADIUS_KM = 6371.0

LANDMARKS_RAD = {
    "jfk": (0.7092964292, -1.2876851706),
    "ewr": (0.7102188594, -1.2944868253),
    "lga": (0.7116969928, -1.2893202334),
    "sol": (0.7101605100, -1.2923203180),
    "nyc": (0.7105961563, -1.2916545533),
}

AIRPORT_KEYS = ("jfk", "ewr", "lga")
AIRPORT_THRESHOLD_KM = 2.0

# Columns the transformer expects (coords in radians, as in the CSV)
RAW_FEATURE_COLUMNS = [
    "Car Condition",
    "Weather",
    "Traffic Condition",
    "passenger_count",
    "month",
    "year",
    "hour",
    "weekday",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]

ENGINEERED_FEATURE_COLUMNS = [
    "Car Condition",
    "Weather",
    "Traffic Condition",
    "passenger_count",
    "month",
    "year",
    "jfk_dist",
    "ewr_dist",
    "lga_dist",
    "sol_dist",
    "nyc_dist",
    "distance",
    "bearing",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "is_airport_trip",
]


def haversine_km(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_rad(lat1, lon1, lat2, lon2):
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return -np.arctan2(x, y)


class TripFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    First step of the sklearn pipeline.

    Input: raw trip DataFrame (radians coords + hour/weekday + categoricals).
    Output: engineered columns used by the ColumnTransformer / model.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"TripFeatureEngineer missing columns: {missing}")

        plat = df["pickup_latitude"].to_numpy(dtype=float)
        plon = df["pickup_longitude"].to_numpy(dtype=float)
        dlat = df["dropoff_latitude"].to_numpy(dtype=float)
        dlon = df["dropoff_longitude"].to_numpy(dtype=float)
        hour = df["hour"].to_numpy(dtype=float)
        weekday = df["weekday"].to_numpy(dtype=float)

        out = pd.DataFrame(
            {
                "Car Condition": df["Car Condition"].values,
                "Weather": df["Weather"].values,
                "Traffic Condition": df["Traffic Condition"].values,
                "passenger_count": df["passenger_count"].to_numpy(dtype=float),
                "month": df["month"].to_numpy(dtype=float),
                "year": df["year"].to_numpy(dtype=float),
                "distance": haversine_km(plat, plon, dlat, dlon),
                "bearing": bearing_rad(plat, plon, dlat, dlon),
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "weekday_sin": np.sin(2 * np.pi * weekday / 7),
                "weekday_cos": np.cos(2 * np.pi * weekday / 7),
                "is_weekend": np.isin(weekday, [5, 6]).astype(int),
            }
        )

        for name, (lat, lon) in LANDMARKS_RAD.items():
            out[f"{name}_dist"] = haversine_km(plat, plon, lat, lon) + haversine_km(
                dlat, dlon, lat, lon
            )

        near = np.zeros(len(df), dtype=bool)
        for key in AIRPORT_KEYS:
            alat, alon = LANDMARKS_RAD[key]
            near |= haversine_km(plat, plon, alat, alon) < AIRPORT_THRESHOLD_KM
            near |= haversine_km(dlat, dlon, alat, alon) < AIRPORT_THRESHOLD_KM
        out["is_airport_trip"] = near.astype(int)

        return out[ENGINEERED_FEATURE_COLUMNS]

    def get_feature_names_out(self, input_features=None):
        return np.array(ENGINEERED_FEATURE_COLUMNS, dtype=object)
