"""
app.py
======
Production Flask application for the Uber Fare Prediction service.

The user provides only raw inputs (map clicks, datetime, passenger count,
and categorical fields). All engineered features are computed server-side
via feature_engineering.create_features() – the same function used at
training time, guaranteeing zero Training/Serving Skew.

Run
---
    python app.py
    # or with gunicorn in production:
    gunicorn -w 4 -b 0.0.0.0:5000 app:app
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from feature_engineering import build_inference_dataframe, validate_raw_input

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & model setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0   # disable static file caching

MODEL_PATH = Path("models/catboost_fare_pipeline.joblib")

try:
    _model = joblib.load(MODEL_PATH)
    print(type(_model))
    logger.info("Model loaded from %s", MODEL_PATH)
except FileNotFoundError:
    logger.error(
        "Model file not found at %s – run train.py first.", MODEL_PATH
    )
    _model = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index() -> str:
    """Render the main prediction form with the Leaflet map."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive raw form inputs, engineer features, and return a fare prediction.

    Accepts both HTML form POST and JSON (for AJAX requests).

    Returns
    -------
    HTML render (form submit) or JSON (fetch / AJAX).
    """
    print("=" * 50)
    print("Predict endpoint reached")
    print("Content-Type:", request.content_type)
    print("JSON:", request.get_json(silent=True))
    print("=" * 50)
    if _model is None:
        msg = "Model is not loaded. Please run train.py first."
        logger.error(msg)
        if request.is_json:
            return jsonify({"error": msg}), 503
        return render_template("result.html", prediction=msg)

    # ── Parse inputs ─────────────────────────────────────────────────────────
    try:
        data = request.get_json(silent=True) or request.form

        pickup_lon_deg  = float(data["pickup_longitude"])
        pickup_lat_deg  = float(data["pickup_latitude"])
        dropoff_lon_deg = float(data["dropoff_longitude"])
        dropoff_lat_deg = float(data["dropoff_latitude"])
        passenger_count = int(data["passenger_count"])
        car_condition   = data["car_condition"]
        weather         = data["weather"]
        traffic_cond    = data["traffic_condition"]
        pickup_datetime = data["pickup_datetime"]

    except (KeyError, ValueError, TypeError) as exc:
        msg = f"Invalid input: {exc}"
        logger.warning(msg)
        if request.is_json:
            return jsonify({"error": msg}), 400
        return render_template("result.html", prediction=msg)

    # ── Validate raw inputs ──────────────────────────────────────────────────
    try:
        validate_raw_input(
            pickup_lon_deg=pickup_lon_deg,
            pickup_lat_deg=pickup_lat_deg,
            dropoff_lon_deg=dropoff_lon_deg,
            dropoff_lat_deg=dropoff_lat_deg,
            passenger_count=passenger_count,
            car_condition=car_condition,
            weather=weather,
            traffic_condition=traffic_cond,
            pickup_datetime=pickup_datetime,  # Pass pickup_datetime for year validation
        )
    except ValueError as exc:
        msg = str(exc)
        logger.warning("Validation failed: %s", msg)
        if request.is_json:
            return jsonify({"error": msg}), 422
        return render_template("result.html", prediction=f"Validation Error: {msg}")

    # ── Feature engineering (same code path as training) ────────────────────
    try:
        print("Building features...")
        X = build_inference_dataframe(
            pickup_lon_deg=pickup_lon_deg,
            pickup_lat_deg=pickup_lat_deg,
            dropoff_lon_deg=dropoff_lon_deg,
            dropoff_lat_deg=dropoff_lat_deg,
            passenger_count=passenger_count,
            car_condition=car_condition,
            weather=weather,
            traffic_condition=traffic_cond,
            pickup_datetime=pickup_datetime,
        )
        print("\n========== FEATURES ==========")
        print(X.T)
        print("==============================\n")
        
    except ValueError as exc:
        msg = f"Feature engineering error: {exc}"
        logger.error(msg)
        if request.is_json:
            return jsonify({"error": msg}), 500
        return render_template("result.html", prediction=msg)

    # ── Prediction ───────────────────────────────────────────────────────────
    try:
        raw_pred = _model.predict(X)[0]
        X_test = X.copy()

        for d in [0.5, 1, 2, 5, 10, 20]:
            X_test["distance"] = d
            print(f"Distance={d} km -> {_model.predict(X_test)[0]:.2f}")
        fare = round(float(raw_pred), 2)
        logger.info(
            "Prediction: $%.2f | pickup=(%.4f, %.4f) dropoff=(%.4f, %.4f)",
            fare,
            pickup_lat_deg, pickup_lon_deg,
            dropoff_lat_deg, dropoff_lon_deg,
        )
    except Exception as exc:
        msg = f"Prediction failed: {exc}"
        logger.exception(msg)
        if request.is_json:
            return jsonify({"error": msg}), 500
        return render_template("result.html", prediction=msg)

    # ── Response ─────────────────────────────────────────────────────────────
    if request.is_json:
        return jsonify({"fare": fare})

    return render_template(
        "result.html",
        prediction=fare,
        pickup_lat=pickup_lat_deg,
        pickup_lon=pickup_lon_deg,
        dropoff_lat=dropoff_lat_deg,
        dropoff_lon=dropoff_lon_deg,
    )


# ---------------------------------------------------------------------------
# Health-check endpoint
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    """Simple liveness / readiness probe."""
    return jsonify({
        "status": "ok",
        "model_loaded": _model is not None,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
