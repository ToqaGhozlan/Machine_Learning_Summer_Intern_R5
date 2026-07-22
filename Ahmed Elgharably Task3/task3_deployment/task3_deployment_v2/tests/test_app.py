"""
tests/test_app.py
==================
Automated smoke tests for the Flask app's request handling (routing, validation,
error messages). These do NOT require the real trained model — they patch in a
tiny dummy pipeline with the exact same input schema, purely to prove the
Flask layer (routes, validation, error handling, JSON/HTML responses) is
correct. They are not a substitute for the real, manual UI screenshots Part D
of the rubric asks for — run the real app with the real
fare_amount_pipeline.joblib for those.

Run:
    pip install pytest scikit-learn pandas numpy joblib
    pytest tests/test_app.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import feature_engineering as fe


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Build a schema-matching dummy pipeline (random data) purely to exercise
    # the Flask routes without needing the real 500k-row dataset.
    numeric_features = [c for c in fe.FEATURE_COLUMNS if c not in ("Car Condition", "Weather", "Traffic Condition")]
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({c: rng.uniform(-1, 1, n) for c in numeric_features})
    df["passenger_count"] = rng.integers(1, 6, n)
    df["year"] = rng.integers(2009, 2016, n)
    df["Car Condition"] = rng.choice(fe.CAR_CONDITIONS, n)
    df["Weather"] = rng.choice(fe.WEATHER_OPTIONS, n)
    df["Traffic Condition"] = rng.choice(fe.TRAFFIC_OPTIONS, n)
    df = df[fe.FEATURE_COLUMNS]
    y = 5 + df["distance"] * 2 + rng.normal(0, 1, n)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("ord", OrdinalEncoder(categories=[fe.CAR_CONDITIONS]), ["Car Condition"]),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ["Weather", "Traffic Condition"]),
    ])
    pipe = Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])
    pipe.fit(df, y)

    model_dir = tmp_path_factory.mktemp("model")
    model_path = model_dir / "fare_amount_pipeline.joblib"
    import joblib
    joblib.dump(pipe, model_path)

    import app as flask_app_module
    flask_app_module.MODEL_PATH = model_path
    flask_app_module.model = pipe
    flask_app_module.model_load_error = None

    flask_app_module.app.config["TESTING"] = True
    return flask_app_module.app.test_client()


VALID_PAYLOAD = dict(
    pickup_datetime="2016-06-15T18:30",
    passenger_count="2",
    pickup_lat="40.7580", pickup_lon="-73.9855",
    dropoff_lat="40.6413", dropoff_lon="-73.7781",
    car_condition="Good", weather="sunny", traffic_condition="Flow Traffic",
)


def test_home_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Uber Fare Predictor" in r.data


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["model_loaded"] is True


def test_valid_prediction_json(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.get_json()
    assert "prediction" in body
    assert body["prediction"] >= 0


def test_valid_prediction_form(client):
    r = client.post("/predict", data=VALID_PAYLOAD)
    assert r.status_code == 200
    assert b"Predicted Fare" in r.data


def test_missing_required_field(client):
    bad = dict(VALID_PAYLOAD)
    bad["passenger_count"] = ""
    r = client.post("/predict", json=bad)
    assert r.status_code == 400
    assert any("is required" in e for e in r.get_json()["errors"])


def test_out_of_range_passenger_count(client):
    bad = dict(VALID_PAYLOAD)
    bad["passenger_count"] = "15"
    r = client.post("/predict", json=bad)
    assert r.status_code == 400
    assert any("Passenger count must be between" in e for e in r.get_json()["errors"])


def test_invalid_datetime(client):
    bad = dict(VALID_PAYLOAD)
    bad["pickup_datetime"] = "not-a-date"
    r = client.post("/predict", json=bad)
    assert r.status_code == 400
    assert any("not a valid date" in e for e in r.get_json()["errors"])


def test_out_of_bounds_coordinates(client):
    bad = dict(VALID_PAYLOAD)
    bad["pickup_lat"] = "5.0"  # nowhere near NYC
    r = client.post("/predict", json=bad)
    assert r.status_code == 400
    assert any("Latitude" in e for e in r.get_json()["errors"])
