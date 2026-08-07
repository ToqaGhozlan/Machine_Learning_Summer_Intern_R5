"""
Uber Fare Prediction — Flask Web Application
Loads the trained pipeline and serves predictions through a web UI.
"""
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Load the trained pipeline ONCE at startup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "uber_fare_pipeline.joblib")
pipeline = joblib.load(PIPELINE_PATH)

# ---------------------------------------------------------------------------
# Constants (must match the training pipeline exactly)
# ---------------------------------------------------------------------------
CAR_CONDITIONS = ["Bad", "Good", "Very Good", "Excellent"]
WEATHER_OPTIONS = ["Windy", "Cloudy", "Stormy", "Sunny", "Rainy"]
TRAFFIC_OPTIONS = ["Flow Traffic", "Dense Traffic", "Congested Traffic"]

# NYC reference point for nyc_dist (optimised from training data)
NYC_REF_LAT = 0.71059625   # radians
NYC_REF_LON = -1.29165511  # radians


# ---------------------------------------------------------------------------
# Helper functions (replicate Task-1/Task-2 feature engineering)
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in km — inputs are in **radians**."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def compute_bearing(lat1, lon1, lat2, lon2):
    """Bearing (radians) from (lat1, lon1) → (lat2, lon2). Inputs in radians."""
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return -np.arctan2(x, y)


def build_feature_row(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat,
                      passenger_count, pickup_dt,
                      car_condition, weather, traffic):
    """
    Turn raw user inputs into the 17-feature DataFrame expected by the
    trained pipeline.

    Coordinate inputs must be in **degrees**; they are converted to radians
    internally (matching the training data scale).
    """
    # Convert degrees → radians (the training data uses radians)
    plat = np.radians(pickup_lat)
    plon = np.radians(pickup_lon)
    dlat = np.radians(dropoff_lat)
    dlon = np.radians(dropoff_lon)

    # Derived features
    distance = haversine(plat, plon, dlat, dlon)
    bearing = compute_bearing(plat, plon, dlat, dlon)
    nyc_dist = haversine(plat, plon, NYC_REF_LAT, NYC_REF_LON) + \
               haversine(dlat, dlon, NYC_REF_LAT, NYC_REF_LON)

    hour = pickup_dt.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    row = {
        "Car Condition": car_condition,
        "Weather": weather,
        "Traffic Condition": traffic,
        "pickup_longitude": plon,
        "pickup_latitude": plat,
        "dropoff_longitude": dlon,
        "dropoff_latitude": dlat,
        "passenger_count": passenger_count,
        "day": pickup_dt.day,
        "month": pickup_dt.month,
        "weekday": pickup_dt.weekday(),
        "year": pickup_dt.year,
        "nyc_dist": nyc_dist,
        "distance": distance,
        "bearing": bearing,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_float(value, name, min_val=None, max_val=None):
    """Return (parsed_float, error_message_or_None)."""
    if value is None or str(value).strip() == "":
        return None, f"{name} is required."
    try:
        v = float(value)
    except ValueError:
        return None, f"{name} must be a number."
    if min_val is not None and v < min_val:
        return None, f"{name} must be at least {min_val}."
    if max_val is not None and v > max_val:
        return None, f"{name} must be at most {max_val}."
    return v, None


def validate_int(value, name, min_val=None, max_val=None):
    if value is None or str(value).strip() == "":
        return None, f"{name} is required."
    try:
        v = int(value)
    except ValueError:
        return None, f"{name} must be a whole number."
    if min_val is not None and v < min_val:
        return None, f"{name} must be at least {min_val}."
    if max_val is not None and v > max_val:
        return None, f"{name} must be at most {max_val}."
    return v, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",
                           car_conditions=CAR_CONDITIONS,
                           weather_options=WEATHER_OPTIONS,
                           traffic_options=TRAFFIC_OPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    errors = []

    # --- Parse & validate every field ---
    pickup_lon, err = validate_float(request.form.get("pickup_lon"), "Pickup Longitude", -180, 180)
    if err: errors.append(err)

    pickup_lat, err = validate_float(request.form.get("pickup_lat"), "Pickup Latitude", -90, 90)
    if err: errors.append(err)

    dropoff_lon, err = validate_float(request.form.get("dropoff_lon"), "Dropoff Longitude", -180, 180)
    if err: errors.append(err)

    dropoff_lat, err = validate_float(request.form.get("dropoff_lat"), "Dropoff Latitude", -90, 90)
    if err: errors.append(err)

    passenger_count, err = validate_int(request.form.get("passenger_count"), "Passenger Count", 1, 6)
    if err: errors.append(err)

    pickup_datetime_str = request.form.get("pickup_datetime", "").strip()
    if not pickup_datetime_str:
        errors.append("Pickup Date & Time is required.")
        pickup_dt = None
    else:
        try:
            pickup_dt = pd.Timestamp(pickup_datetime_str)
        except Exception:
            errors.append("Pickup Date & Time is not a valid date/time.")
            pickup_dt = None

    car_condition = request.form.get("car_condition", "").strip()
    if car_condition not in CAR_CONDITIONS:
        errors.append("Please select a valid Car Condition.")

    weather = request.form.get("weather", "").strip()
    if weather not in WEATHER_OPTIONS:
        errors.append("Please select a valid Weather condition.")

    traffic = request.form.get("traffic", "").strip()
    if traffic not in TRAFFIC_OPTIONS:
        errors.append("Please select a valid Traffic Condition.")

    # --- If any errors, re-render form with messages ---
    if errors:
        return render_template("index.html",
                               car_conditions=CAR_CONDITIONS,
                               weather_options=WEATHER_OPTIONS,
                               traffic_options=TRAFFIC_OPTIONS,
                               errors=errors,
                               form=request.form)

    # --- Build features & predict ---
    try:
        features = build_feature_row(
            pickup_lon, pickup_lat, dropoff_lon, dropoff_lat,
            passenger_count, pickup_dt,
            car_condition, weather, traffic,
        )
        prediction = pipeline.predict(features)[0]
        prediction = round(float(prediction), 2)
    except Exception as e:
        errors.append(f"Prediction failed: {e}")
        return render_template("index.html",
                               car_conditions=CAR_CONDITIONS,
                               weather_options=WEATHER_OPTIONS,
                               traffic_options=TRAFFIC_OPTIONS,
                               errors=errors,
                               form=request.form)

    return render_template("index.html",
                           car_conditions=CAR_CONDITIONS,
                           weather_options=WEATHER_OPTIONS,
                           traffic_options=TRAFFIC_OPTIONS,
                           prediction=prediction,
                           form=request.form)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
