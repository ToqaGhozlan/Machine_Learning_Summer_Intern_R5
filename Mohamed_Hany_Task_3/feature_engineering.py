"""
feature_engineering.py
=======================
Single source of truth for all preprocessing.

VALIDATED FORMULAS (from validate_formulas.py):
  distance : Haversine pickup→dropoff  MAE=0.000000 on original dataset
  bearing  : arctan2(y,x) with dlon = lon1-lon2 (pickup-dropoff)  MAE=0.000107
  jfk_dist : Haversine pickup→JFK (real coords, retrained model)
  lga_dist : Haversine pickup→LGA (real coords, retrained model)
  nyc_dist : Haversine pickup→NYC center (real coords, retrained model)

Used identically during training (train.py) and inference (app.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
EARTH_RADIUS_KM: float = 6_371.0

# Real geographic reference points (degrees → radians)
_JFK_LAT = np.deg2rad(40.6413)   # JFK Airport
_JFK_LON = np.deg2rad(-73.7781)

_LGA_LAT = np.deg2rad(40.7769)   # LaGuardia Airport
_LGA_LON = np.deg2rad(-73.8740)

_NYC_LAT = np.deg2rad(40.7128)   # NYC city center (Manhattan)
_NYC_LON = np.deg2rad(-74.0060)

# Ordinal maps (identical to training notebook)
_CAR_MAP: dict[str, int] = {
    "Bad": 0, "Good": 1, "Very Good": 2, "Excellent": 3,
}
_TRAFFIC_MAP: dict[str, int] = {
    "Flow Traffic": 0, "Dense Traffic": 1, "Congested Traffic": 2,
}

# Final feature column order — must match ColumnTransformer in train.py
FEATURE_COLUMNS: list[str] = [
    "Car Condition",
    "Traffic Condition",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "jfk_dist",
    "lga_dist",
    "nyc_dist",
    "distance",
    "bearing",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "rush_hour",
    "is_weekend",
    "is_night",
    "Weather",
]

# ── Internal helpers ─────────────────────────────────────────────────────────

def _haversine(
    lat1: "np.ndarray | float",
    lon1: "np.ndarray | float",
    lat2: float,
    lon2: float,
) -> "np.ndarray | float":
    """Great-circle distance in km. All args in radians."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _bearing(
    lat1: "np.ndarray | float",
    lon1: "np.ndarray | float",
    lat2: "np.ndarray | float",
    lon2: "np.ndarray | float",
) -> "np.ndarray | float":
    """
    Bearing from point-1 to point-2, in radians.
    VALIDATED: dlon = lon1 - lon2  (pickup minus dropoff) → MAE=0.000107
    """
    dlon = lon1 - lon2
    y = np.sin(dlon) * np.cos(lat2)
    x = (np.cos(lat1) * np.sin(lat2)
         - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    return np.arctan2(y, x)


def _cyclic(series: "pd.Series", period: float,
            offset: float = 0.0) -> "tuple[pd.Series, pd.Series]":
    angle = 2.0 * np.pi * (series - offset) / period
    return np.sin(angle), np.cos(angle)


# ── Public API ───────────────────────────────────────────────────────────────

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw DataFrame → model-ready feature matrix.

    Input columns (all in radians for coordinates):
      pickup_longitude, pickup_latitude,
      dropoff_longitude, dropoff_latitude,
      passenger_count, Weather, Car Condition, Traffic Condition,
      pickup_datetime  (or pre-split: hour, weekday, month, year)

    Returns DataFrame with exactly FEATURE_COLUMNS in that order.
    """
    df = df.copy()

    # ── 1. Parse datetime ────────────────────────────────────────────────────
    if "pickup_datetime" in df.columns:
        dt = pd.to_datetime(df["pickup_datetime"])
        df["year"]    = dt.dt.year
        df["month"]   = dt.dt.month
        df["weekday"] = dt.dt.weekday   # 0=Mon … 6=Sun
        df["hour"]    = dt.dt.hour

    # ── 2. Coordinates (already radians) ────────────────────────────────────
    lat1 = df["pickup_latitude"]
    lon1 = df["pickup_longitude"]
    lat2 = df["dropoff_latitude"]
    lon2 = df["dropoff_longitude"]

    # ── 3. Trip distance & bearing ───────────────────────────────────────────
    df["distance"] = _haversine(lat1, lon1, lat2, lon2)
    df["bearing"]  = _bearing(lat1, lon1, lat2, lon2)

    # ── 4. Airport / city distances ──────────────────────────────────────────
    df["jfk_dist"] = _haversine(lat1, lon1, _JFK_LAT, _JFK_LON)
    df["lga_dist"] = _haversine(lat1, lon1, _LGA_LAT, _LGA_LON)
    df["nyc_dist"] = _haversine(lat1, lon1, _NYC_LAT, _NYC_LON)

    # ── 5. Cyclic temporal features ──────────────────────────────────────────
    df["hour_sin"],    df["hour_cos"]    = _cyclic(df["hour"],    24.0)
    df["weekday_sin"], df["weekday_cos"] = _cyclic(df["weekday"], 7.0)
    df["month_sin"],   df["month_cos"]   = _cyclic(df["month"],   12.0, offset=1.0)

    # ── 6. Binary flags ──────────────────────────────────────────────────────
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype("int8")
    df["rush_hour"]  = (
        ((df["hour"] >= 7)  & (df["hour"] <= 9))
        | ((df["hour"] >= 17) & (df["hour"] <= 22))
    ).astype("int8")
    df["is_night"] = (
        (df["hour"] >= 23) | (df["hour"] <= 5)
    ).astype("int8")

    # ── 7. Ordinal encoding ──────────────────────────────────────────────────
    df["Car Condition"] = (
        df["Car Condition"].map(_CAR_MAP).fillna(-1).astype("int8")
    )
    df["Traffic Condition"] = (
        df["Traffic Condition"].map(_TRAFFIC_MAP).fillna(-1).astype("int8")
    )

    # ── 8. Select & order ────────────────────────────────────────────────────
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after transform: {missing}")

    return df[FEATURE_COLUMNS]


def build_inference_dataframe(
    pickup_lon_deg: float,
    pickup_lat_deg: float,
    dropoff_lon_deg: float,
    dropoff_lat_deg: float,
    passenger_count: int,
    car_condition: str,
    weather: str,
    traffic_condition: str,
    pickup_datetime: str,
) -> pd.DataFrame:
    """
    Entry point for Flask inference.
    Accepts raw user inputs in DEGREES, converts to radians, calls create_features().
    """
    raw = pd.DataFrame([{
        "pickup_longitude":  np.deg2rad(pickup_lon_deg),
        "pickup_latitude":   np.deg2rad(pickup_lat_deg),
        "dropoff_longitude": np.deg2rad(dropoff_lon_deg),
        "dropoff_latitude":  np.deg2rad(dropoff_lat_deg),
        "passenger_count":   passenger_count,
        "Car Condition":     car_condition,
        "Traffic Condition": traffic_condition,
        "Weather":           weather,
        "pickup_datetime":   pickup_datetime,
    }])
    return create_features(raw)


def validate_raw_input(
    pickup_lon_deg: float, pickup_lat_deg: float,
    dropoff_lon_deg: float, dropoff_lat_deg: float,
    passenger_count: int, car_condition: str,
    weather: str, traffic_condition: str,
    pickup_datetime: str = None,
) -> None:
    """Validate raw user inputs. Raises ValueError on invalid data."""
    if pickup_datetime:
        # Validate that pickup_datetime year is within the training period
        dt = pd.to_datetime(pickup_datetime)
        if not (2009 <= dt.year <= 2015):
            raise ValueError("pickup_datetime must be between years 2009 and 2015.")
            
    if car_condition not in _CAR_MAP:
        raise ValueError(f"car_condition must be one of {set(_CAR_MAP)}")
    if traffic_condition not in _TRAFFIC_MAP:
        raise ValueError(f"traffic_condition must be one of {set(_TRAFFIC_MAP)}")
    if weather.lower() not in {"sunny", "cloudy", "windy", "stormy"}:
        raise ValueError("weather must be one of {sunny, cloudy, windy, stormy}")
    if not (1 <= passenger_count <= 8):
        raise ValueError("passenger_count must be between 1 and 8")
    for name, val, lo, hi in [
        ("pickup_longitude",  pickup_lon_deg,  -74.30, -73.65),
        ("pickup_latitude",   pickup_lat_deg,   40.45,  40.95),
        ("dropoff_longitude", dropoff_lon_deg, -74.30, -73.65),
        ("dropoff_latitude",  dropoff_lat_deg,  40.45,  40.95),
    ]:
        if not (lo <= val <= hi):
            raise ValueError(
                f"{name}={val:.4f} outside NYC bounds [{lo}, {hi}]"
            )
