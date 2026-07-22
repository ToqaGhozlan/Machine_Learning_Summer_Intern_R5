"""
feature_engineering.py
-----------------------
Reproduces, step by step, the SAME feature engineering pipeline used in the
training notebook (Uber_Fare_Prediction.ipynb) so that raw values submitted
through the Flask form are transformed exactly the same way as the training
data before being handed to the saved model pipeline.

VERIFIED against final_internship_data.csv (the actual training data), to
floating-point precision (residuals ~1e-12, i.e. exact, not approximate):
- Earth radius used in the haversine formula is 6371 km (confirmed by an
  exact match, to the last decimal, against the dataset's own `distance`
  column).
- jfk_dist / ewr_dist / lga_dist / sol_dist / nyc_dist = the SUM of the
  haversine distance from the pickup point to the landmark PLUS the haversine
  distance from the dropoff point to the landmark (not the min of the two).
  The exact landmark coordinates below were recovered via least-squares
  fitting against thousands of rows of the real dataset.
- bearing = the initial bearing from pickup to dropoff, computed with
  atan2, left in RADIANS (not converted to degrees) and with the sign
  flipped relative to the standard formula, i.e.
  bearing = -atan2(sin(dlon)*cos(lat2), cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon))
- hour/day/month/year/weekday all match plain pandas datetime attributes
  (weekday: Monday=0 ... Sunday=6, i.e. pandas' Timestamp.weekday()).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Reference points used to compute landmark distances (lat, lon in degrees)
# ---------------------------------------------------------------------------
LANDMARKS = {
    "jfk_dist": (40.639722, -73.778889),   # JFK Airport
    "ewr_dist": (40.692500, -74.168611),   # Newark Airport
    "lga_dist": (40.777250, -73.872611),   # LaGuardia Airport
    "sol_dist": (40.689200, -74.044500),   # Statue of Liberty
    "nyc_dist": (40.714167, -74.006389),   # NYC center (Manhattan / Battery Park)
}

# NYC bounding box used for basic sanity-checking of submitted coordinates
NYC_LAT_MIN, NYC_LAT_MAX = 40.4774, 40.9176
NYC_LON_MIN, NYC_LON_MAX = -74.2591, -73.7004

# Ordinal encoding maps (must match the notebook exactly)
CAR_CONDITION_ORDER = {"Bad": 0, "Good": 1, "Very Good": 2, "Excellent": 3}
TRAFFIC_ORDER = {"Flow Traffic": 0, "Dense Traffic": 1, "Congested Traffic": 2}

# Weather categories (one-hot, "Cloudy" is the dropped baseline -> all 0s)
WEATHER_OPTIONS = ["Cloudy", "Rainy", "Stormy", "Sunny", "Windy"]
WEATHER_DUMMY_COLS = ["rainy", "stormy", "sunny", "windy"]

# Columns that received a log1p transform during training (skew > 1.0)
LOG1P_COLUMNS = ["passenger_count", "jfk_dist", "ewr_dist", "nyc_dist", "calculated_distance"]

# Final column order expected by the saved pipeline's ColumnTransformer
FINAL_COLUMNS = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "passenger_count", "hour", "day", "month", "year",
    "jfk_dist", "ewr_dist", "lga_dist", "sol_dist", "nyc_dist", "bearing",
    "calculated_distance", "car_cond_enc", "weekday_sin", "weekday_cos",
    "Traffic_Encoded", "rainy", "stormy", "sunny", "windy",
]


class InputValidationError(Exception):
    """Raised when submitted trip data is missing, malformed, or out of range."""
    pass


def _haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km. Inputs already in radians."""
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def _initial_bearing_rad(lon1, lat1, lon2, lat2):
    """
    Initial bearing from point 1 to point 2, in RADIANS, sign-flipped and
    left unwrapped (range approx. -pi to pi). This matches the convention
    verified against the `bearing` column in final_internship_data.csv —
    do NOT convert to degrees or wrap to 0-360, that does not match training.
    Inputs already in radians.
    """
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return -np.arctan2(x, y)


def validate_raw_input(form):
    """
    Validates the raw dict of submitted form values.
    Returns a cleaned dict of typed values, or raises InputValidationError
    with a human-readable message describing the first problem found.
    """
    errors = []
    cleaned = {}

    required_fields = [
        "pickup_datetime", "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude", "passenger_count",
        "car_condition", "weather", "traffic_condition",
    ]
    for field in required_fields:
        if not form.get(field, "").strip():
            errors.append(f"'{field.replace('_', ' ')}' is required.")

    if errors:
        raise InputValidationError(" ".join(errors))

    # --- datetime ---
    try:
        dt = pd.to_datetime(form["pickup_datetime"])
    except Exception:
        raise InputValidationError("Pickup date/time could not be understood. Use the date/time picker.")
    cleaned["hour"] = dt.hour
    cleaned["day"] = dt.day
    cleaned["month"] = dt.month
    cleaned["year"] = dt.year
    cleaned["weekday"] = dt.weekday()  # Monday=0 ... Sunday=6

    # --- coordinates ---
    coord_fields = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    for field in coord_fields:
        try:
            cleaned[field] = float(form[field])
        except ValueError:
            raise InputValidationError(f"'{field.replace('_', ' ')}' must be a number.")

    if not (NYC_LAT_MIN <= cleaned["pickup_latitude"] <= NYC_LAT_MAX):
        raise InputValidationError("Pickup latitude is outside the plausible NYC range (40.48 to 40.92).")
    if not (NYC_LON_MIN <= cleaned["pickup_longitude"] <= NYC_LON_MAX):
        raise InputValidationError("Pickup longitude is outside the plausible NYC range (-74.26 to -73.70).")
    if not (NYC_LAT_MIN <= cleaned["dropoff_latitude"] <= NYC_LAT_MAX):
        raise InputValidationError("Dropoff latitude is outside the plausible NYC range (40.48 to 40.92).")
    if not (NYC_LON_MIN <= cleaned["dropoff_longitude"] <= NYC_LON_MAX):
        raise InputValidationError("Dropoff longitude is outside the plausible NYC range (-74.26 to -73.70).")

    # --- passenger count ---
    try:
        passenger_count = int(form["passenger_count"])
    except ValueError:
        raise InputValidationError("Passenger count must be a whole number.")
    if not (1 <= passenger_count <= 6):
        raise InputValidationError("Passenger count must be between 1 and 6.")
    cleaned["passenger_count"] = passenger_count

    # --- categorical fields ---
    car_condition = form["car_condition"].strip()
    if car_condition not in CAR_CONDITION_ORDER:
        raise InputValidationError(f"Car condition must be one of: {', '.join(CAR_CONDITION_ORDER)}.")
    cleaned["car_condition"] = car_condition

    weather = form["weather"].strip()
    if weather not in WEATHER_OPTIONS:
        raise InputValidationError(f"Weather must be one of: {', '.join(WEATHER_OPTIONS)}.")
    cleaned["weather"] = weather

    traffic_condition = form["traffic_condition"].strip()
    if traffic_condition not in TRAFFIC_ORDER:
        raise InputValidationError(f"Traffic condition must be one of: {', '.join(TRAFFIC_ORDER)}.")
    cleaned["traffic_condition"] = traffic_condition

    return cleaned


def engineer_features(cleaned):
    """
    Takes the cleaned/typed raw input (from validate_raw_input) and returns a
    single-row pandas DataFrame with exactly the columns/order the trained
    pipeline expects, having applied the same encodings and transforms used
    during training.
    """
    # 1. Coordinates: degrees -> radians (matches notebook cell that converts
    #    any value outside the radian range; here raw form input is always in
    #    degrees, so we convert directly).
    pickup_lon_rad = np.radians(cleaned["pickup_longitude"])
    pickup_lat_rad = np.radians(cleaned["pickup_latitude"])
    dropoff_lon_rad = np.radians(cleaned["dropoff_longitude"])
    dropoff_lat_rad = np.radians(cleaned["dropoff_latitude"])

    # 2. Trip distance via haversine (matches notebook's calculated_distance)
    calculated_distance = _haversine_km(pickup_lon_rad, pickup_lat_rad, dropoff_lon_rad, dropoff_lat_rad)

    # 3. Bearing pickup -> dropoff (radians, sign-flipped — see module docstring)
    bearing = _initial_bearing_rad(pickup_lon_rad, pickup_lat_rad, dropoff_lon_rad, dropoff_lat_rad)

    # 4. Landmark distances: SUM of (pickup -> landmark) and (dropoff -> landmark)
    #    haversine distances — verified against final_internship_data.csv.
    landmark_distances = {}
    for col, (lm_lat, lm_lon) in LANDMARKS.items():
        lm_lat_rad, lm_lon_rad = np.radians(lm_lat), np.radians(lm_lon)
        d_pickup = _haversine_km(pickup_lon_rad, pickup_lat_rad, lm_lon_rad, lm_lat_rad)
        d_dropoff = _haversine_km(dropoff_lon_rad, dropoff_lat_rad, lm_lon_rad, lm_lat_rad)
        landmark_distances[col] = d_pickup + d_dropoff

    # 5. Ordinal encodings
    car_cond_enc = CAR_CONDITION_ORDER[cleaned["car_condition"]]
    traffic_encoded = TRAFFIC_ORDER[cleaned["traffic_condition"]]

    # 6. One-hot weather (Cloudy = baseline -> all dummy cols 0)
    weather_dummies = {c: 0 for c in WEATHER_DUMMY_COLS}
    weather_key = cleaned["weather"].lower()
    if weather_key in weather_dummies:
        weather_dummies[weather_key] = 1

    # 7. Cyclical weekday encoding
    weekday_sin = np.sin(2 * np.pi * cleaned["weekday"] / 7)
    weekday_cos = np.cos(2 * np.pi * cleaned["weekday"] / 7)

    row = {
        "pickup_longitude": pickup_lon_rad,
        "pickup_latitude": pickup_lat_rad,
        "dropoff_longitude": dropoff_lon_rad,
        "dropoff_latitude": dropoff_lat_rad,
        "passenger_count": cleaned["passenger_count"],
        "hour": cleaned["hour"],
        "day": cleaned["day"],
        "month": cleaned["month"],
        "year": cleaned["year"],
        "jfk_dist": landmark_distances["jfk_dist"],
        "ewr_dist": landmark_distances["ewr_dist"],
        "lga_dist": landmark_distances["lga_dist"],
        "sol_dist": landmark_distances["sol_dist"],
        "nyc_dist": landmark_distances["nyc_dist"],
        "bearing": bearing,
        "calculated_distance": calculated_distance,
        "car_cond_enc": car_cond_enc,
        "weekday_sin": weekday_sin,
        "weekday_cos": weekday_cos,
        "Traffic_Encoded": traffic_encoded,
        **weather_dummies,
    }

    # 8. Apply the same log1p transform used in training to the skewed columns
    for col in LOG1P_COLUMNS:
        row[col] = np.log1p(row[col])

    df = pd.DataFrame([row])
    return df[FINAL_COLUMNS]


def preprocess_and_predict(form, pipeline):
    """
    Full request -> prediction flow:
    validate -> engineer features -> predict (log scale) -> invert log -> USD.
    Raises InputValidationError on bad input (caller should catch and display).
    """
    cleaned = validate_raw_input(form)
    X = engineer_features(cleaned)
    y_pred_log = pipeline.predict(X)[0]
    fare_usd = float(np.expm1(y_pred_log))
    return fare_usd, X
