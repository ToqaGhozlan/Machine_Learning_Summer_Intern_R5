from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from uber_fare_pipeline import build_raw_payload, validate_trip_payload


def test_valid_payload_passes_validation():
    payload = build_raw_payload(
        pickup_datetime="2024-06-10T08:30",
        pickup_longitude=-73.9857,
        pickup_latitude=40.7580,
        dropoff_longitude=-73.9712,
        dropoff_latitude=40.7831,
        passenger_count=1,
        car_condition="Good",
        weather="sunny",
        traffic_conditions="Flow Traffic",
    )

    is_valid, errors = validate_trip_payload(payload)
    assert is_valid is True
    assert errors == []


def test_invalid_payload_fails_validation():
    payload = build_raw_payload(
        pickup_datetime="not-a-date",
        pickup_longitude=500,
        pickup_latitude=40.7580,
        dropoff_longitude=-73.9712,
        dropoff_latitude=40.7831,
        passenger_count=0,
        car_condition="Good",
        weather="sunny",
        traffic_conditions="Flow Traffic",
    )

    is_valid, errors = validate_trip_payload(payload)
    assert is_valid is False
    assert errors

