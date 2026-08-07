"""
feature_engineering.py
=======================
Single source of truth for turning raw trip input into the exact feature row
`fare_amount_pipeline.joblib` (Task 2) was trained on. Used identically by
app.py at inference time — there is only one copy of this logic, so training
and serving cannot silently drift apart.

Trained feature set (must match Task 2's `numeric_features` / `ordinal_features`
/ `onehot_features` exactly):
    passenger_count, year, nyc_dist, distance,
    hour_sin, hour_cos, weekday_sin, weekday_cos, month_sin, month_cos,
    bearing_sin, bearing_cos, is_weekend,
    'Car Condition' (ordinal), 'Weather' (one-hot), 'Traffic Condition' (one-hot)

Formula notes
-------------
`distance`, `bearing`, and `nyc_dist` are *pre-engineered* columns that already
existed in final_internship_data.csv before Task 2 — nobody on this project
computed them from raw coordinates originally, so the exact formula/unit used
to generate them isn't directly documented. Two independent signals point to
kilometres with Earth radius 6371 (rather than miles):

  1. The Task 1 data dictionary explicitly labels `Distance` as "(km)".
  2. A classmate's independent reproduction (validate_formulas.py-style check
     against the real dataset) got MAE ~0.000000 for `distance` and ~0.000107
     for `bearing` using R=6371 km and bearing = atan2(y, x) with
     dlon = pickup_lon − dropoff_lon.

Both are used below. **Run `validate_formulas.py` against your own copy of
final_internship_data.csv once** (it has the raw coordinates *and* the
pre-engineered columns together) to confirm this on your exact data before
trusting it in production — see that script for how to flip the assumption if
your MAE comes out lower with miles or the opposite bearing sign.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

# ── Validated constants ──────────────────────────────────────────────────────
EARTH_RADIUS_KM: float = 6371.0

# Reference point used for `nyc_dist` (distance to NYC / Manhattan center)
NYC_LAT: float = 40.7128
NYC_LON: float = -74.0060

# Winsorizing cap applied to `distance` in Task 2 (same fixed threshold, same units)
DISTANCE_CAP: float = 100.0

# Categories the pipeline was trained on — must match Task 2 exactly
CAR_CONDITIONS: list[str] = ["Bad", "Good", "Very Good", "Excellent"]
WEATHER_OPTIONS: list[str] = ["sunny", "cloudy", "rainy", "windy", "stormy"]
TRAFFIC_OPTIONS: list[str] = ["Flow Traffic", "Dense Traffic", "Congested Traffic"]

# Sensible input bounds for validation (NYC metro area + plausible ride values)
LAT_BOUNDS = (39.5, 41.5)
LON_BOUNDS = (-75.0, -72.5)
PASSENGER_BOUNDS = (1, 6)

FEATURE_COLUMNS: list[str] = [
    "passenger_count", "year", "nyc_dist", "distance",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos", "bearing_sin", "bearing_cos", "is_weekend",
    "Car Condition", "Weather", "Traffic Condition",
]


# ── Geo formulas (validated — see module docstring) ─────────────────────────
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points (degrees in)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Bearing from point 1 to point 2, in radians (-pi..pi).
    Validated convention: dlon = lon1 - lon2 (pickup minus dropoff).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon1 - lon2)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return math.atan2(y, x)


def _cyclical(value: float, period: float) -> tuple[float, float]:
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


# ── Validation ────────────────────────────────────────────────────────────────
def validate_raw_input(form: dict) -> list[str]:
    """Validate raw form input. Returns a list of human-readable error strings (empty = valid)."""
    errors: list[str] = []

    required = [
        "pickup_datetime", "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon",
        "passenger_count", "car_condition", "weather", "traffic_condition",
    ]
    for field in required:
        if not str(form.get(field, "")).strip():
            errors.append(f"'{field.replace('_', ' ')}' is required.")
    if errors:
        return errors

    try:
        pickup_lat = float(form["pickup_lat"])
        pickup_lon = float(form["pickup_lon"])
        dropoff_lat = float(form["dropoff_lat"])
        dropoff_lon = float(form["dropoff_lon"])
    except (ValueError, TypeError):
        errors.append("Pickup/dropoff coordinates must be numbers.")
        pickup_lat = pickup_lon = dropoff_lat = dropoff_lon = None

    try:
        passenger_count = int(form["passenger_count"])
    except (ValueError, TypeError):
        errors.append("Passenger count must be a whole number.")
        passenger_count = None

    try:
        pickup_dt = datetime.strptime(form["pickup_datetime"], "%Y-%m-%dT%H:%M")
    except (ValueError, TypeError):
        errors.append("Pickup date/time is not a valid date/time.")
        pickup_dt = None

    if errors:
        return errors

    if not (LAT_BOUNDS[0] <= pickup_lat <= LAT_BOUNDS[1]) or not (LAT_BOUNDS[0] <= dropoff_lat <= LAT_BOUNDS[1]):
        errors.append(f"Latitude looks out of range for the NYC area ({LAT_BOUNDS[0]} to {LAT_BOUNDS[1]}).")
    if not (LON_BOUNDS[0] <= pickup_lon <= LON_BOUNDS[1]) or not (LON_BOUNDS[0] <= dropoff_lon <= LON_BOUNDS[1]):
        errors.append(f"Longitude looks out of range for the NYC area ({LON_BOUNDS[0]} to {LON_BOUNDS[1]}).")
    if passenger_count is not None and not (PASSENGER_BOUNDS[0] <= passenger_count <= PASSENGER_BOUNDS[1]):
        errors.append(f"Passenger count must be between {PASSENGER_BOUNDS[0]} and {PASSENGER_BOUNDS[1]}.")
    if form.get("car_condition") not in CAR_CONDITIONS:
        errors.append("Car condition must be one of: " + ", ".join(CAR_CONDITIONS))
    if form.get("weather") not in WEATHER_OPTIONS:
        errors.append("Weather must be one of: " + ", ".join(WEATHER_OPTIONS))
    if form.get("traffic_condition") not in TRAFFIC_OPTIONS:
        errors.append("Traffic condition must be one of: " + ", ".join(TRAFFIC_OPTIONS))

    return errors


# ── Public API ────────────────────────────────────────────────────────────────
def build_feature_row(form: dict) -> pd.DataFrame:
    """
    Turn validated raw form input into the single-row DataFrame the trained
    pipeline expects (see FEATURE_COLUMNS). Call validate_raw_input() first —
    this function assumes the input is already valid.
    """
    pickup_lat = float(form["pickup_lat"])
    pickup_lon = float(form["pickup_lon"])
    dropoff_lat = float(form["dropoff_lat"])
    dropoff_lon = float(form["dropoff_lon"])
    passenger_count = int(form["passenger_count"])
    pickup_dt = datetime.strptime(form["pickup_datetime"], "%Y-%m-%dT%H:%M")

    distance = min(haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon), DISTANCE_CAP)
    nyc_dist = haversine_km(pickup_lat, pickup_lon, NYC_LAT, NYC_LON)
    bearing = bearing_rad(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    hour, weekday, month, year = pickup_dt.hour, pickup_dt.weekday(), pickup_dt.month, pickup_dt.year
    hour_sin, hour_cos = _cyclical(hour, 24)
    weekday_sin, weekday_cos = _cyclical(weekday, 7)
    month_sin, month_cos = _cyclical(month, 12)

    row = {
        "passenger_count": passenger_count,
        "year": year,
        "nyc_dist": nyc_dist,
        "distance": distance,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "weekday_sin": weekday_sin, "weekday_cos": weekday_cos,
        "month_sin": month_sin, "month_cos": month_cos,
        "bearing_sin": math.sin(bearing), "bearing_cos": math.cos(bearing),
        "is_weekend": int(weekday >= 5),
        "Car Condition": form["car_condition"],
        "Weather": form["weather"],
        "Traffic Condition": form["traffic_condition"],
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
