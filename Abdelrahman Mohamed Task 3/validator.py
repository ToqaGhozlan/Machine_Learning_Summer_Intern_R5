from datetime import datetime

LAT_MIN, LAT_MAX = 38.9, 42.4
LON_MIN, LON_MAX = -77.4, -71.6

MIN_PASSENGERS, MAX_PASSENGERS = 1, 6

# this class raise a message when submitted trip details are missing or out of range
class ValidationError(Exception):

    pass #means do nothing ,without it Python gives an error because every class or function must contain at least one statement

def _require(payload, field):
    if field not in payload or payload[field] in (None, "", []):
        raise ValidationError(f"'{field}' is required.")
    return payload[field]


def _as_float(value, field):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'{field}' must be a number.")


def _validate_point(payload, prefix):
    lat = _as_float(_require(payload, f"{prefix}_lat"), f"{prefix}_lat")
    lon = _as_float(_require(payload, f"{prefix}_lon"), f"{prefix}_lon")
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        raise ValidationError(
            f"The {prefix} point looks outside the New York City area "
            f"the model was trained on. Please pick a point on the map."
        )
    return lat, lon

# validates the inputs of the user ----------------------------------------------
def validate_trip_details(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValidationError("Malformed request.")

    pickup_lat, pickup_lon = _validate_point(payload, "pickup")
    dropoff_lat, dropoff_lon = _validate_point(payload, "dropoff")

    if pickup_lat == dropoff_lat and pickup_lon == dropoff_lon:
        raise ValidationError(
            "Pickup and drop-off points are identical. Please choose two "
            "different locations on the map."
        )

    passenger_raw = _require(payload, "passenger_count")
    try:
        passenger_count = int(passenger_raw)
    except (TypeError, ValueError):
        raise ValidationError("'passenger_count' must be a whole number.")
    if not (MIN_PASSENGERS <= passenger_count <= MAX_PASSENGERS):
        raise ValidationError(
            f"Passenger count must be between {MIN_PASSENGERS} and {MAX_PASSENGERS}."
        )

    date_str = _require(payload, "date")
    time_str = _require(payload, "time")
    try:
        datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        raise ValidationError(
            "Date/time is invalid. Expected date as YYYY-MM-DD and time as HH:MM."
        )

    return {
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "dropoff_lat": dropoff_lat,
        "dropoff_lon": dropoff_lon,
        "passenger_count": passenger_count,
        "date": date_str,
        "time": time_str,
    }
