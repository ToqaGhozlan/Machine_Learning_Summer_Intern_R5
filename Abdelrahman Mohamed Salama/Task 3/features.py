import math
from datetime import datetime

EARTH_RADIUS_KM = 6371.0

LANDMARKS_DEG = {
    "jfk_dist": (40.6413, -73.7781),  
    "ewr_dist": (40.6895, -74.1745),   
    "lga_dist": (40.7769, -73.8740),   
    "sol_dist": (40.6892, -74.0445),  
    "nyc_dist": (40.7128, -74.0060),   
}

RUSH_HOURS = {7, 8, 9, 16, 17, 18, 19}


def _to_rad(deg: float) -> float:
    return math.radians(deg)


def haversine_km(lat1_rad, lon1_rad, lat2_rad, lon2_rad) -> float:
    """Great-circle distance in km. Inputs must already be in radians,
    matching how the training data was engineered."""
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return EARTH_RADIUS_KM * c


def bearing_rad(lat1_rad, lon1_rad, lat2_rad, lon2_rad) -> float:
    """Initial compass bearing from point 1 to point 2, in radians,
    range (-pi, pi] -- matches the sign/range seen in the training data's
    `bearing` column."""
    dlon = lon2_rad - lon1_rad
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad)
         - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon))
    return math.atan2(x, y)


def compute_geo_features(pickup_lat_deg, pickup_lon_deg,
                          dropoff_lat_deg, dropoff_lon_deg):
    """
    Takes plain lat/lon in degrees (what Leaflet gives us) and returns a dict
    with:
      - the radian coordinates the model was trained on
      - the engineered distance / bearing / landmark columns the model needs
      - a couple of extra human-readable numbers for the results panel
    """
    pu_lat_r, pu_lon_r = _to_rad(pickup_lat_deg), _to_rad(pickup_lon_deg)
    do_lat_r, do_lon_r = _to_rad(dropoff_lat_deg), _to_rad(dropoff_lon_deg)

    trip_distance_km = haversine_km(pu_lat_r, pu_lon_r, do_lat_r, do_lon_r)
    bearing = bearing_rad(pu_lat_r, pu_lon_r, do_lat_r, do_lon_r)
    bearing_deg = (math.degrees(bearing) + 360) % 360

    landmark_features = {}
    pickup_to_center_km = dropoff_to_center_km = 0.0
    for col, (lm_lat_deg, lm_lon_deg) in LANDMARKS_DEG.items():
        lm_lat_r, lm_lon_r = _to_rad(lm_lat_deg), _to_rad(lm_lon_deg)
        d_pickup = haversine_km(pu_lat_r, pu_lon_r, lm_lat_r, lm_lon_r)
        d_dropoff = haversine_km(do_lat_r, do_lon_r, lm_lat_r, lm_lon_r)
        landmark_features[col] = d_pickup + d_dropoff
        if col == "nyc_dist":
            pickup_to_center_km = d_pickup
            dropoff_to_center_km = d_dropoff

    return {
        # model inputs (radians) these go straight into the pipeline
        "pickup_latitude": pu_lat_r,
        "pickup_longitude": pu_lon_r,
        "dropoff_latitude": do_lat_r,
        "dropoff_longitude": do_lon_r,
        "distance": trip_distance_km,
        "bearing": bearing,
        **landmark_features,
        "bearing_deg": bearing_deg,
        "pickup_to_center_km": pickup_to_center_km,
        "dropoff_to_center_km": dropoff_to_center_km,
    }

# takes date 'YYYY-MM-DD', time 'HH:MM' and returns (hour/day/month/weekday/year) plus (rush hour, weekend, weekday name)
def compute_time_features(date_str: str, time_str: str):
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    weekday = dt.weekday()  # Monday=0 ... Sunday=6
    return {
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "weekday": weekday,
        "year": dt.year,
        # display-only extras
        "weekday_name": dt.strftime("%A"),
        "is_rush_hour": dt.hour in RUSH_HOURS,
        "is_weekend": weekday >= 5,
    }
