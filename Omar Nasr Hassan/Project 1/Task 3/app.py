"""
Uber Fare Prediction - Flask App
Loads the trained pipeline once at startup, and for every user-submitted trip
recreates the exact same features used in Task 2 (distance, landmark distances,
bearing, date/time parts, is_airport_trip) before calling the model.
"""

from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained pipeline ONCE when the app starts (not per request)
# ---------------------------------------------------------------------------
MODEL_PATH = "uber_fare_pipeline_small.pkl"
model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------------
# Fixed landmark coordinates (used to compute jfk_dist / ewr_dist / lga_dist),
# same landmarks used to build these columns in the original dataset.
# ---------------------------------------------------------------------------
JFK_COORDS = (40.6413, -73.7781)
EWR_COORDS = (40.6895, -74.1745)
LGA_COORDS = (40.7769, -73.8740)

AIRPORT_RADIUS_KM = 2  # same threshold used in Task 2's is_airport_trip flag


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/long points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Initial compass bearing (0-360 degrees) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def validate_inputs(form):
    """Check required fields are present, numeric, and within sensible ranges.
    Returns (cleaned_values_dict, error_message). error_message is None if valid."""

    errors = []
    values = {}

    # --- Numeric fields with sensible range checks ---
    numeric_fields = {
        "pickup_longitude": (-74.3, -72.9),
        "pickup_latitude": (40.5, 41.8),
        "dropoff_longitude": (-74.3, -72.9),
        "dropoff_latitude": (40.5, 41.8),
        "passenger_count": (1, 6),
    }

    for field, (low, high) in numeric_fields.items():
        raw = form.get(field, "").strip()
        if raw == "":
            errors.append(f"'{field.replace('_', ' ')}' is required.")
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"'{field.replace('_', ' ')}' must be a number.")
            continue
        if not (low <= val <= high):
            errors.append(f"'{field.replace('_', ' ')}' must be between {low} and {high}.")
            continue
        values[field] = val

    # --- Date/time field ---
    raw_datetime = form.get("pickup_datetime", "").strip()
    if raw_datetime == "":
        errors.append("Pickup date/time is required.")
    else:
        try:
            values["pickup_datetime"] = datetime.strptime(raw_datetime, "%Y-%m-%dT%H:%M")
        except ValueError:
            errors.append("Pickup date/time format is invalid.")

    # --- Categorical fields ---
    valid_car_conditions = ["Bad", "Good", "Very Good", "Excellent"]
    valid_weather = ["sunny", "cloudy", "rainy", "windy", "stormy"]
    valid_traffic = ["Congested Traffic", "Dense Traffic", "Flow Traffic"]

    car_condition = form.get("car_condition", "")
    if car_condition not in valid_car_conditions:
        errors.append("Please select a valid car condition.")
    else:
        values["car_condition"] = car_condition

    weather = form.get("weather", "")
    if weather not in valid_weather:
        errors.append("Please select a valid weather condition.")
    else:
        values["weather"] = weather

    traffic_condition = form.get("traffic_condition", "")
    if traffic_condition not in valid_traffic:
        errors.append("Please select a valid traffic condition.")
    else:
        values["traffic_condition"] = traffic_condition

    if errors:
        return None, " ".join(errors)
    return values, None


def build_feature_row(values):
    """Recreate every engineered feature exactly as done in Task 2,
    from the raw user-submitted values."""

    pickup_lat = values["pickup_latitude"]
    pickup_lon = values["pickup_longitude"]
    dropoff_lat = values["dropoff_latitude"]
    dropoff_lon = values["dropoff_longitude"]
    dt = values["pickup_datetime"]

    distance = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    bearing = calculate_bearing(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

    jfk_dist = haversine(pickup_lat, pickup_lon, *JFK_COORDS)
    ewr_dist = haversine(pickup_lat, pickup_lon, *EWR_COORDS)
    lga_dist = haversine(pickup_lat, pickup_lon, *LGA_COORDS)

    is_airport_trip = int(
        (jfk_dist < AIRPORT_RADIUS_KM)
        or (ewr_dist < AIRPORT_RADIUS_KM)
        or (lga_dist < AIRPORT_RADIUS_KM)
    )

    row = {
        "car_condition": values["car_condition"],
        "weather": values["weather"],
        "traffic_condition": values["traffic_condition"],
        "passenger_count": values["passenger_count"],
        "hour": dt.hour,
        "day_of_month": dt.day,
        "month": dt.month,
        "weekday": dt.weekday(),
        "year": dt.year,
        "jfk_dist": jfk_dist,
        "ewr_dist": ewr_dist,
        "lga_dist": lga_dist,
        "distance": distance,
        "bearing": bearing,
        "is_airport_trip": is_airport_trip,
    }

    return pd.DataFrame([row])


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    values, error = validate_inputs(request.form)

    if error:
        return render_template("index.html", error=error, form=request.form)

    try:
        X_input = build_feature_row(values)
        prediction = model.predict(X_input)[0]
        prediction = round(float(prediction), 2)
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Something went wrong while predicting: {e}",
            form=request.form,
        )

    return render_template("index.html", prediction=prediction, form=request.form)


if __name__ == "__main__":
    app.run(debug=True)
