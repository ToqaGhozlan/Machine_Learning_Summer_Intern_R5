"""
app.py
======
Flask application for the Uber Fare Prediction service (Task 3).

Loads fare_amount_pipeline.joblib (Task 2's fitted sklearn Pipeline — scaler +
encoders + model all bundled together) ONCE at startup. Raw user input from the
form is turned into the exact training-time feature row by
feature_engineering.build_feature_row(), so training and serving stay in sync
by construction (single source of truth, see feature_engineering.py).

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

from feature_engineering import (
    CAR_CONDITIONS,
    TRAFFIC_OPTIONS,
    WEATHER_OPTIONS,
    build_feature_row,
    validate_raw_input,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "fare_amount_pipeline.joblib"
model = None
model_load_error: str | None = None
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as exc:  # file missing / incompatible sklearn version / etc.
    model_load_error = str(exc)
    logger.error("Could not load model from %s: %s", MODEL_PATH, exc)


def _base_context(**extra):
    return dict(
        car_conditions=CAR_CONDITIONS,
        weather_options=WEATHER_OPTIONS,
        traffic_options=TRAFFIC_OPTIONS,
        model_load_error=model_load_error,
        **extra,
    )


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", **_base_context())


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts an HTML form POST (page render) or a JSON body (AJAX / API)."""
    is_ajax = request.is_json
    form = request.get_json(silent=True) if is_ajax else request.form

    if model is None:
        msg = f"Model could not be loaded on the server: {model_load_error}"
        logger.error(msg)
        if is_ajax:
            return jsonify({"errors": [msg]}), 503
        return render_template("index.html", **_base_context(errors=[msg], form_values=form)), 503

    errors = validate_raw_input(form)
    if errors:
        logger.warning("Validation failed: %s", errors)
        if is_ajax:
            return jsonify({"errors": errors}), 400
        return render_template("index.html", **_base_context(errors=errors, form_values=form)), 400

    try:
        row = build_feature_row(form)
        pred = max(0.0, float(model.predict(row)[0]))
    except Exception as exc:
        msg = f"The model could not produce a prediction: {exc}"
        logger.exception(msg)
        if is_ajax:
            return jsonify({"errors": [msg]}), 500
        return render_template("index.html", **_base_context(errors=[msg], form_values=form)), 500

    prediction = round(pred, 2)
    distance = round(float(row["distance"].iloc[0]), 2)
    logger.info("Prediction: $%.2f (distance=%.2f km)", prediction, distance)

    if is_ajax:
        return jsonify({"prediction": prediction, "computed_distance": distance})

    return render_template(
        "index.html",
        **_base_context(prediction=prediction, computed_distance=distance, form_values=form),
    )


@app.route("/health")
def health():
    """Liveness / readiness probe."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    app.run(debug=True)
