from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, flash, render_template, request

from uber_fare_pipeline import (
    NYC_AIRPORTS,
    build_raw_payload,
    haversine_km,
    bearing_degrees,
    validate_trip_payload,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "uber_fare_model.pkl"

app = Flask(__name__)
app.secret_key = "uber-fare-secret-key"


def load_model():
    if not MODEL_PATH.exists():
        return None, f"Model file not found at {MODEL_PATH.name}. Run train_and_export.py first."
    return joblib.load(MODEL_PATH), None


MODEL, MODEL_ERROR = load_model()

# ---------------------------------------------------------------------------
# Geocoding helper -- Nominatim (OpenStreetMap, free, no API key)
# ---------------------------------------------------------------------------
def geocode_address(address: str) -> tuple[str, str] | tuple[None, None]:
    """Convert a free-text address to (latitude, longitude) in **degrees**.

    Returns (lat, lon) as strings on success, or (None, None) on failure.
    """
    if not address or not address.strip():
        return None, None
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address.strip(),
        "format": "json",
        "limit": 1,
    }
    headers = {
        "User-Agent": "UberFarePredictor/1.0 (flask-app)",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            return data[0]["lat"], data[0]["lon"]
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Coordinate conversion: degrees  ->  radians
#
# The training data (final_internship_data.csv) stores coordinates in radians.
# For example, pickup_longitude values are around -1.2888 rad (~ -73.84 deg).
# The saved pipeline was fitted with these radian values (StandardScaler + model).
# Geocoders such as Nominatim return coordinates in degrees, so we convert
# here to match the training-data distribution.
# ---------------------------------------------------------------------------
def deg_to_rad(value: float) -> float:
    """Convert a degree value to radians."""
    return np.radians(value)


# ---------------------------------------------------------------------------
# Build the full model payload with all pre-computed features
# ---------------------------------------------------------------------------
def build_model_payload(
    pickup_datetime_str: str,
    pickup_lat_deg: float,
    pickup_lon_deg: float,
    dropoff_lat_deg: float,
    dropoff_lon_deg: float,
    passenger_count: int,
    car_condition: str,
    weather: str,
    traffic_conditions: str,
) -> dict:
    """Build the payload expected by the saved pipeline.

    The pipeline's TripFeatureEngineer computes time features (Hour, Day, Month,
    Week, IsWeekend) and geometry features (Distance, Bearing, airport distances)
    from the raw columns -- but those raw coordinate columns must be in **radians**
    to match training data.  The haversine_km and bearing_degrees helpers call
    np.radians() expecting degrees, so if TripFeatureEngineer tried to recompute
    Distance/Bearing from radian coordinates it would produce wrong values.

    To avoid this we pre-compute all derived features from the *degree* coordinates
    and include them in the payload.  TripFeatureEngineer checks ``if "Distance"
    not in frame.columns`` etc. and skips any column that is already present.
    """
    # --- time features ----------------------------------------------------
    pickup_dt = pd.to_datetime(pickup_datetime_str)
    hour = pickup_dt.hour
    day_of_week = pickup_dt.dayofweek   # Monday=0, Sunday=6
    month = pickup_dt.month
    week = pickup_dt.isocalendar().week
    is_weekend = 1 if day_of_week >= 5 else 0   # Sat=5, Sun=6

    # --- geometry features from DEGREE coordinates ------------------------
    pickup_lat_s = pd.Series([pickup_lat_deg], name="lat")
    pickup_lon_s = pd.Series([pickup_lon_deg], name="lon")
    dropoff_lat_s = pd.Series([dropoff_lat_deg], name="lat")
    dropoff_lon_s = pd.Series([dropoff_lon_deg], name="lon")

    distance_km = float(
        haversine_km(pickup_lat_s, pickup_lon_s, float(dropoff_lat_deg), float(dropoff_lon_deg)).iloc[0]
    )
    bearing = float(
        bearing_degrees(pickup_lat_s, pickup_lon_s, dropoff_lat_s, dropoff_lon_s).iloc[0]
    )

    airport_distances: dict[str, float] = {}
    for feature_name, (airport_lat, airport_lon) in NYC_AIRPORTS.items():
        airport_distances[feature_name] = float(
            haversine_km(pickup_lat_s, pickup_lon_s, airport_lat, airport_lon).iloc[0]
        )

    # --- coordinates in RADIANS to match training data --------------------
    pickup_lat_rad = deg_to_rad(pickup_lat_deg)
    pickup_lon_rad = deg_to_rad(pickup_lon_deg)
    dropoff_lat_rad = deg_to_rad(dropoff_lat_deg)
    dropoff_lon_rad = deg_to_rad(dropoff_lon_deg)

    return {
        "Pickup_Datetime": pickup_datetime_str,
        "Pickup_Longitude": pickup_lon_rad,
        "Pickup_Latitude": pickup_lat_rad,
        "Dropoff_Longitude": dropoff_lon_rad,
        "Dropoff_Latitude": dropoff_lat_rad,
        "Passenger_Count": passenger_count,
        "Car_Condition": car_condition,
        "Weather": weather,
        "Traffic_Conditions": traffic_conditions,
        # Pre-computed time features (TripFeatureEngineer skips if present)
        "Hour": hour,
        "Day": day_of_week,
        "Month": month,
        "Week": int(week),
        "IsWeekend": is_weekend,
        # Pre-computed geometry features (TripFeatureEngineer skips if present)
        "Distance": distance_km,
        "Bearing": bearing,
        **airport_distances,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    defaults = {
        "name": "",
        "day": 15,
        "month": 6,
        "year": 2024,
        "time": "08:30",
        "pickup_address": "Times Square, New York, NY",
        "dropoff_address": "Central Park, New York, NY",
        "passenger_count": 1,
        "car_condition": "Good",
        "weather": "sunny",
        "traffic_conditions": "Flow Traffic",
    }
    return render_template(
        "index.html",
        defaults=defaults,
        prediction=None,
        errors=[],
        model_ready=MODEL is not None,
        model_error=MODEL_ERROR,
    )


@app.route("/predict", methods=["POST"])
def predict():
    # ---- 1. Extract friendly form fields ---------------------------------
    name = request.form.get("name", "").strip()
    day = request.form.get("day", "").strip()
    month = request.form.get("month", "").strip()
    year = request.form.get("year", "").strip()
    time_str = request.form.get("time", "").strip()
    pickup_address = request.form.get("pickup_address", "").strip()
    dropoff_address = request.form.get("dropoff_address", "").strip()
    passenger_count_raw = request.form.get("passenger_count", "").strip()
    car_condition = request.form.get("car_condition", "").strip()
    weather = request.form.get("weather", "").strip()
    traffic_conditions = request.form.get("traffic_conditions", "").strip()

    # Defaults for re-rendering the form on error
    defaults = {
        "name": name,
        "day": day,
        "month": month,
        "year": year,
        "time": time_str,
        "pickup_address": pickup_address,
        "dropoff_address": dropoff_address,
        "passenger_count": passenger_count_raw,
        "car_condition": car_condition,
        "weather": weather,
        "traffic_conditions": traffic_conditions,
    }

    errors: list[str] = []

    # ---- 2. Validate friendly fields -------------------------------------
    if not name:
        errors.append("Name is required.")

    # Date parts
    try:
        d = int(day)
        m = int(month)
        y = int(year)
        if not (1 <= d <= 31):
            errors.append("Day must be between 1 and 31.")
        if not (1 <= m <= 12):
            errors.append("Month must be between 1 and 12.")
        if not (1900 <= y <= 2100):
            errors.append("Year must be between 1900 and 2100.")
    except (TypeError, ValueError):
        errors.append("Day, Month, and Year must be valid integers.")

    if not time_str:
        errors.append("Time is required (HH:MM format).")

    if not pickup_address:
        errors.append("Pickup address is required.")
    if not dropoff_address:
        errors.append("Dropoff address is required.")

    try:
        pc = int(passenger_count_raw)
        if not (1 <= pc <= 8):
            errors.append("Passenger count must be between 1 and 8.")
    except (TypeError, ValueError):
        errors.append("Passenger count must be an integer.")

    # Categorical values (must match training data categories)
    valid_car_conditions = {"Bad", "Excellent", "Good", "Very Good"}
    if car_condition not in valid_car_conditions:
        errors.append(f"Car condition must be one of: {', '.join(sorted(valid_car_conditions))}.")

    valid_weather = {"cloudy", "rainy", "stormy", "sunny", "windy"}
    if weather not in valid_weather:
        errors.append(f"Weather must be one of: {', '.join(sorted(valid_weather))}.")

    valid_traffic = {"Congested Traffic", "Dense Traffic", "Flow Traffic"}
    if traffic_conditions not in valid_traffic:
        errors.append(f"Traffic conditions must be one of: {', '.join(sorted(valid_traffic))}.")

    if errors:
        return render_template(
            "index.html",
            defaults=defaults,
            prediction=None,
            errors=errors,
            model_ready=MODEL is not None,
            model_error=MODEL_ERROR,
        ), 400

    # ---- 3. Build pickup datetime string ---------------------------------
    pickup_datetime_str = f"{y:04d}-{m:02d}-{d:02d}T{time_str}"

    # ---- 4. Geocode addresses (returns degrees) --------------------------
    pickup_lat, pickup_lon = geocode_address(pickup_address)
    dropoff_lat, dropoff_lon = geocode_address(dropoff_address)

    if pickup_lat is None:
        errors.append(
            f"Could not geocode pickup address '{pickup_address}'. "
            "Please provide a more specific location (e.g. street + city or landmark)."
        )
    if dropoff_lat is None:
        errors.append(
            f"Could not geocode dropoff address '{dropoff_address}'. "
            "Please provide a more specific location (e.g. street + city or landmark)."
        )
    if errors:
        return render_template(
            "index.html",
            defaults=defaults,
            prediction=None,
            errors=errors,
            model_ready=MODEL is not None,
            model_error=MODEL_ERROR,
        ), 400

    pickup_lat_deg = float(pickup_lat)
    pickup_lon_deg = float(pickup_lon)
    dropoff_lat_deg = float(dropoff_lat)
    dropoff_lon_deg = float(dropoff_lon)

    # ---- 5. Build full model payload with pre-computed features ----------
    model_payload = build_model_payload(
        pickup_datetime_str=pickup_datetime_str,
        pickup_lat_deg=pickup_lat_deg,
        pickup_lon_deg=pickup_lon_deg,
        dropoff_lat_deg=dropoff_lat_deg,
        dropoff_lon_deg=dropoff_lon_deg,
        passenger_count=pc,
        car_condition=car_condition,
        weather=weather,
        traffic_conditions=traffic_conditions,
    )

    # ---- 6. Validate the raw payload columns -----------------------------
    raw_payload = build_raw_payload(
        pickup_datetime=model_payload["Pickup_Datetime"],
        pickup_longitude=model_payload["Pickup_Longitude"],
        pickup_latitude=model_payload["Pickup_Latitude"],
        dropoff_longitude=model_payload["Dropoff_Longitude"],
        dropoff_latitude=model_payload["Dropoff_Latitude"],
        passenger_count=model_payload["Passenger_Count"],
        car_condition=model_payload["Car_Condition"],
        weather=model_payload["Weather"],
        traffic_conditions=model_payload["Traffic_Conditions"],
    )

    valid, payload_errors = validate_trip_payload(raw_payload)
    if not valid:
        flash("Please fix the highlighted input issues and try again.", "error")
        return render_template(
            "index.html",
            defaults=defaults,
            prediction=None,
            errors=payload_errors,
            model_ready=MODEL is not None,
            model_error=MODEL_ERROR,
        ), 400

    # ---- 7. Check model availability -------------------------------------
    if MODEL is None:
        flash("The model artifact is missing. Train and save uber_fare_model.pkl first.", "error")
        return render_template(
            "index.html",
            defaults=defaults,
            prediction=None,
            errors=[MODEL_ERROR or "Model is not loaded."],
            model_ready=False,
            model_error=MODEL_ERROR,
        ), 500

    # ---- 8. Predict ------------------------------------------------------
    payload_df = pd.DataFrame([model_payload])
    try:
        prediction = float(MODEL.predict(payload_df)[0])
    except Exception as exc:  # pragma: no cover - runtime safety
        flash("Prediction failed because the saved model could not process the input.", "error")
        return render_template(
            "index.html",
            defaults=defaults,
            prediction=None,
            errors=[str(exc)],
            model_ready=True,
            model_error=MODEL_ERROR,
        ), 500

    return render_template(
        "index.html",
        defaults=defaults,
        prediction=round(prediction, 2),
        errors=[],
        model_ready=True,
        model_error=None,
        # Pass geocoded degree coordinates and addresses for the map
        pickup_lat=pickup_lat_deg,
        pickup_lon=pickup_lon_deg,
        dropoff_lat=dropoff_lat_deg,
        dropoff_lon=dropoff_lon_deg,
        pickup_address=pickup_address,
        dropoff_address=dropoff_address,
    )


if __name__ == "__main__":
    app.run(debug=True)

