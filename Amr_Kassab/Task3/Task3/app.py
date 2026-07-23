"""
Flask app for Uber fare prediction (Task 3 — Parts C & D).

Loads the fitted pipeline once at startup. Feature engineering lives inside
the pipeline (TripFeatureEngineer); this file only validates the form and
builds the raw column DataFrame the pipeline expects.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

# Import before joblib.load so the custom pipeline step can unpickle
from trip_transformer import RAW_FEATURE_COLUMNS, TripFeatureEngineer  # noqa: F401

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_pipeline.joblib"

VALID_CAR = ["Bad", "Good", "Very Good", "Excellent"]
VALID_TRAFFIC = ["Flow Traffic", "Dense Traffic", "Congested Traffic"]
VALID_WEATHER = ["sunny", "rainy", "cloudy", "stormy", "windy"]

app = Flask(__name__)
app.secret_key = "uber-fare-task3-dev-key"

print(f"Loading model from {MODEL_PATH} …")
model = joblib.load(MODEL_PATH)
print("Model loaded. Steps:", list(model.regressor_.named_steps.keys()))


def _default_form_values():
    return {
        "pickup_latitude": "40.7580",
        "pickup_longitude": "-73.9855",
        "dropoff_latitude": "40.6413",
        "dropoff_longitude": "-73.7781",
        "passenger_count": "1",
        "pickup_datetime": "2014-06-15T18:30",
        "car_condition": "Good",
        "weather": "sunny",
        "traffic_condition": "Dense Traffic",
    }


def validate_and_build_raw(form: dict[str, str]):
    """Validate UI input and return a one-row DataFrame of RAW_FEATURE_COLUMNS (radians)."""
    errors: list[str] = []
    required = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "passenger_count",
        "pickup_datetime",
        "car_condition",
        "weather",
        "traffic_condition",
    ]
    for field in required:
        if str(form.get(field, "")).strip() == "":
            errors.append(f"'{field.replace('_', ' ').title()}' is required.")
    if errors:
        return None, errors

    try:
        pickup_lat = float(form["pickup_latitude"])
        pickup_lon = float(form["pickup_longitude"])
        dropoff_lat = float(form["dropoff_latitude"])
        dropoff_lon = float(form["dropoff_longitude"])
    except ValueError:
        return None, ["Coordinates must be valid numbers (decimal degrees)."]

    for label, lat, lon in (
        ("Pickup", pickup_lat, pickup_lon),
        ("Dropoff", dropoff_lat, dropoff_lon),
    ):
        if not (40.0 <= lat <= 41.5):
            errors.append(f"{label} latitude must be between 40.0 and 41.5 (NYC area).")
        if not (-75.0 <= lon <= -72.5):
            errors.append(f"{label} longitude must be between -75.0 and -72.5 (NYC area).")

    try:
        passengers = int(float(form["passenger_count"]))
    except ValueError:
        return None, ["Passenger count must be a whole number."]
    if passengers < 1 or passengers > 6:
        errors.append("Passenger count must be between 1 and 6.")

    dt_raw = form["pickup_datetime"].strip()
    pickup_dt = None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            pickup_dt = datetime.strptime(dt_raw, fmt)
            break
        except ValueError:
            continue
    if pickup_dt is None:
        errors.append("Pickup date/time could not be parsed. Use the date-time picker.")

    car = form["car_condition"].strip()
    weather = form["weather"].strip()
    traffic = form["traffic_condition"].strip()
    if car not in VALID_CAR:
        errors.append(f"Car condition must be one of: {', '.join(VALID_CAR)}.")
    if weather not in VALID_WEATHER:
        errors.append(f"Weather must be one of: {', '.join(VALID_WEATHER)}.")
    if traffic not in VALID_TRAFFIC:
        errors.append(f"Traffic condition must be one of: {', '.join(VALID_TRAFFIC)}.")

    if errors or pickup_dt is None:
        return None, errors

    # Degrees (UI) -> radians (pipeline / dataset convention)
    plat, plon = float(np.radians(pickup_lat)), float(np.radians(pickup_lon))
    dlat, dlon = float(np.radians(dropoff_lat)), float(np.radians(dropoff_lon))

    # Rough distance check (same spirit as Task 2 cleaning)
    dlat_r, dlon_r = dlat - plat, dlon - plon
    a = np.sin(dlat_r / 2) ** 2 + np.cos(plat) * np.cos(dlat) * np.sin(dlon_r / 2) ** 2
    dist = float(2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))
    if dist <= 0:
        errors.append("Pickup and dropoff appear to be the same location (distance = 0).")
    elif dist >= 500:
        errors.append("Trip distance is unrealistically large (>= 500 km). Check coordinates.")
    if errors:
        return None, errors

    row = pd.DataFrame(
        [
            {
                "Car Condition": car,
                "Weather": weather,
                "Traffic Condition": traffic,
                "passenger_count": passengers,
                "month": pickup_dt.month,
                "year": pickup_dt.year,
                "hour": pickup_dt.hour,
                "weekday": pickup_dt.weekday(),
                "pickup_latitude": plat,
                "pickup_longitude": plon,
                "dropoff_latitude": dlat,
                "dropoff_longitude": dlon,
            }
        ],
        columns=RAW_FEATURE_COLUMNS,
    )
    return row, []


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        form_data=_default_form_values(),
        car_options=VALID_CAR,
        weather_options=VALID_WEATHER,
        traffic_options=VALID_TRAFFIC,
        prediction=None,
        engineered=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_data = {key: request.form.get(key, "") for key in _default_form_values()}
    raw, errors = validate_and_build_raw(form_data)

    if errors:
        for err in errors:
            flash(err, "error")
        return render_template(
            "index.html",
            form_data=form_data,
            car_options=VALID_CAR,
            weather_options=VALID_WEATHER,
            traffic_options=VALID_TRAFFIC,
            prediction=None,
            engineered=None,
        ), 400

    fare = float(model.predict(raw)[0])
    # Peek at engineered features via the pipeline's first step (for the result panel)
    eng = model.regressor_.named_steps["engineer"].transform(raw)
    engineered = {
        "distance_km": round(float(eng["distance"].iloc[0]), 3),
        "bearing": round(float(eng["bearing"].iloc[0]), 4),
        "is_airport_trip": int(eng["is_airport_trip"].iloc[0]),
        "is_weekend": int(eng["is_weekend"].iloc[0]),
        "month": int(eng["month"].iloc[0]),
        "year": int(eng["year"].iloc[0]),
    }

    return render_template(
        "index.html",
        form_data=form_data,
        car_options=VALID_CAR,
        weather_options=VALID_WEATHER,
        traffic_options=VALID_TRAFFIC,
        prediction=round(fare, 2),
        engineered=engineered,
    )


@app.route("/reset", methods=["GET"])
def reset():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
