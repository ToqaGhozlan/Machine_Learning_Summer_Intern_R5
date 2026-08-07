# turns validated trip details into the exact feature row the saved in the pkl file
import pandas as pd

from features import compute_geo_features, compute_time_features

MODEL_COLUMNS = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "passenger_count", "hour", "day", "month", "weekday", "year",
    "jfk_dist", "ewr_dist", "lga_dist", "sol_dist", "nyc_dist",
    "distance", "bearing",
    "Car Condition", "Weather", "Traffic Condition",
]


# those are conditions of the future trip that a fare-estimate form can't 
# know in advance We fall back to fixed, neutral values that were present
DEFAULT_CAR_CONDITION = "Very Good"
DEFAULT_WEATHER = "cloudy"
DEFAULT_TRAFFIC_CONDITION = "Flow Traffic"

# trip: the clean dict returned by sidebar.validate_trip_details()
def build_feature_row(trip: dict) -> pd.DataFrame:
    geo = compute_geo_features( #geo features----------------------------------------
        trip["pickup_lat"], trip["pickup_lon"],
        trip["dropoff_lat"], trip["dropoff_lon"],
    )
    time_feats = compute_time_features(trip["date"], trip["time"]) # time features----------------------------------

    row = {
        "pickup_longitude": geo["pickup_longitude"],
        "pickup_latitude": geo["pickup_latitude"],
        "dropoff_longitude": geo["dropoff_longitude"],
        "dropoff_latitude": geo["dropoff_latitude"],
        "passenger_count": trip["passenger_count"],
        "hour": time_feats["hour"],
        "day": time_feats["day"],
        "month": time_feats["month"],
        "weekday": time_feats["weekday"],
        "year": time_feats["year"],
        "jfk_dist": geo["jfk_dist"],
        "ewr_dist": geo["ewr_dist"],
        "lga_dist": geo["lga_dist"],
        "sol_dist": geo["sol_dist"],
        "nyc_dist": geo["nyc_dist"],
        "distance": geo["distance"],
        "bearing": geo["bearing"],
        "Car Condition": DEFAULT_CAR_CONDITION,
        "Weather": DEFAULT_WEATHER,
        "Traffic Condition": DEFAULT_TRAFFIC_CONDITION,
    }
    df = pd.DataFrame([row], columns=MODEL_COLUMNS)
    return df, geo, time_feats



    """Runs the full pipeline on validated trip details and returns
    a JSON-friendly dict describing the prediction, ready for the results"""
def predict_fare(model, trip: dict) -> dict:
    df, geo, time_feats = build_feature_row(trip)#call build function-----------------------------------
    fare = float(model.predict(df)[0])
    fare = max(fare, 0.0)

    return {
        "predicted_fare": round(fare, 2),
        "trip_distance_km": round(geo["distance"], 2),
        "direction_deg": round(geo["bearing_deg"]),
        "day": time_feats["weekday_name"],
        "month_year": f"{time_feats['month']}/{time_feats['year']}",
        "is_rush_hour": time_feats["is_rush_hour"],
        "hour": f"{time_feats['hour']:02d}:00",
        "is_weekend": time_feats["is_weekend"],
        "pickup_to_center_km": round(geo["pickup_to_center_km"], 2),
        "dropoff_to_center_km": round(geo["dropoff_to_center_km"], 2),
    }
