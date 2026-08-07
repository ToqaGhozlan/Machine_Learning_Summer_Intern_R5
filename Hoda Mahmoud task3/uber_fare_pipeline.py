from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


NYC_AIRPORTS: dict[str, tuple[float, float]] = {
    # Airport coordinates in DEGREES (lat, lon).
    # The haversine/bearing helpers below convert degrees -> radians internally,
    # so these values must be kept in degrees.
    "JFK_Dist": (40.6413, -73.7781),
    "EWR_Dist": (40.6895, -74.1745),
    "LGA_Dist": (40.7769, -73.8740),
    "SOL_Dist": (40.6892, -74.0445),
}


RAW_REQUIRED_COLUMNS = [
    "Pickup_Datetime",
    "Pickup_Longitude",
    "Pickup_Latitude",
    "Dropoff_Longitude",
    "Dropoff_Latitude",
    "Passenger_Count",
    "Car_Condition",
    "Weather",
    "Traffic_Conditions",
]


def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: float, lon2: float) -> pd.Series:
    """Return great-circle distance in kilometers."""
    radius = 6371.0
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return radius * 2.0 * np.arcsin(np.sqrt(a))


def bearing_degrees(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Calculate initial bearing in degrees from point 1 to point 2."""
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    dlon = lon2_rad - lon1_rad
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


class TripFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create the same trip features for training and inference."""

    def __init__(self, include_time_features: bool = True, include_airport_features: bool = True) -> None:
        self.include_time_features = include_time_features
        self.include_airport_features = include_airport_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        # Use getattr for backward compatibility with pickled model instances
        # that lack the include_time_features attribute.
        inc_time = getattr(self, "include_time_features", True)
        required_columns = [
            column
            for column in RAW_REQUIRED_COLUMNS
            if inc_time or column != "Pickup_Datetime"
        ]
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        if inc_time:
            frame["Pickup_Datetime"] = pd.to_datetime(frame["Pickup_Datetime"], errors="coerce")
            if frame["Pickup_Datetime"].isna().any():
                raise ValueError("Pickup_Datetime contains invalid values.")

        for column in [
            "Pickup_Longitude",
            "Pickup_Latitude",
            "Dropoff_Longitude",
            "Dropoff_Latitude",
            "Passenger_Count",
        ]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        # When raw datetime is available, derive Task2-style time and trip features on the fly.
        if "Pickup_Datetime" in frame.columns:
            pickup_time = pd.to_datetime(frame["Pickup_Datetime"], errors="coerce")
            if pickup_time.isna().any():
                raise ValueError("Pickup_Datetime contains invalid values.")

            pickup_lat = frame["Pickup_Latitude"]
            pickup_lon = frame["Pickup_Longitude"]
            dropoff_lat = frame["Dropoff_Latitude"]
            dropoff_lon = frame["Dropoff_Longitude"]

            if "Hour" not in frame.columns:
                frame["Hour"] = pickup_time.dt.hour.astype(int)
            if "Day" not in frame.columns:
                frame["Day"] = pickup_time.dt.dayofweek.astype(int)
            if "Month" not in frame.columns:
                frame["Month"] = pickup_time.dt.month.astype(int)
            if "Week" not in frame.columns:
                frame["Week"] = pickup_time.dt.isocalendar().week.astype(int)
            if "Distance" not in frame.columns:
                frame["Distance"] = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
            if "Bearing" not in frame.columns:
                frame["Bearing"] = bearing_degrees(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)

            if self.include_airport_features:
                for feature_name, (airport_lat, airport_lon) in NYC_AIRPORTS.items():
                    if feature_name not in frame.columns:
                        frame[feature_name] = haversine_km(pickup_lat, pickup_lon, airport_lat, airport_lon)

            frame = frame.drop(columns=["Pickup_Datetime"])

        return frame


@dataclass(frozen=True)
class ValidationLimits:
    passenger_min: int = 1
    passenger_max: int = 8
    longitude_min: float = -180.0
    longitude_max: float = 180.0
    latitude_min: float = -90.0
    latitude_max: float = 90.0


def validate_trip_payload(payload: dict) -> tuple[bool, list[str]]:
    """Validate raw form input before prediction."""
    limits = ValidationLimits()
    errors: list[str] = []

    for column in RAW_REQUIRED_COLUMNS:
        if payload.get(column) in (None, ""):
            errors.append(f"{column} is required.")

    if errors:
        return False, errors

    try:
        passenger_count = int(payload["Passenger_Count"])
        if not (limits.passenger_min <= passenger_count <= limits.passenger_max):
            errors.append("Passenger_Count must be between 1 and 8.")
    except (TypeError, ValueError):
        errors.append("Passenger_Count must be an integer.")

    for field in ["Pickup_Longitude", "Dropoff_Longitude"]:
        try:
            value = float(payload[field])
            if not (limits.longitude_min <= value <= limits.longitude_max):
                errors.append(f"{field} must be a valid longitude.")
        except (TypeError, ValueError):
            errors.append(f"{field} must be numeric.")

    for field in ["Pickup_Latitude", "Dropoff_Latitude"]:
        try:
            value = float(payload[field])
            if not (limits.latitude_min <= value <= limits.latitude_max):
                errors.append(f"{field} must be a valid latitude.")
        except (TypeError, ValueError):
            errors.append(f"{field} must be numeric.")

    try:
        pd.to_datetime(payload["Pickup_Datetime"])
    except Exception:
        errors.append("Pickup_Datetime must be a valid date and time.")

    # Categories must match the training data in final_internship_data.csv
    allowed_categories = {
        "Car_Condition": {"Bad", "Excellent", "Good", "Very Good"},
        "Weather": {"cloudy", "rainy", "stormy", "sunny", "windy"},
        "Traffic_Conditions": {"Congested Traffic", "Dense Traffic", "Flow Traffic"},
    }
    for field, allowed_values in allowed_categories.items():
        if payload[field] not in allowed_values:
            errors.append(f"{field} must be one of: {', '.join(sorted(allowed_values))}.")

    return len(errors) == 0, errors


def build_raw_payload(
    pickup_datetime: str,
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    passenger_count: int,
    car_condition: str,
    weather: str,
    traffic_conditions: str,
) -> dict:
    return {
        "Pickup_Datetime": pickup_datetime,
        "Pickup_Longitude": pickup_longitude,
        "Pickup_Latitude": pickup_latitude,
        "Dropoff_Longitude": dropoff_longitude,
        "Dropoff_Latitude": dropoff_latitude,
        "Passenger_Count": passenger_count,
        "Car_Condition": car_condition,
        "Weather": weather,
        "Traffic_Conditions": traffic_conditions,
    }