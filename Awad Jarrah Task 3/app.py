"""
app.py
------
Flask web app for the Uber Fare Prediction model.

Run locally with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.

The trained pipeline (preprocessing + model) is loaded ONCE at startup from
`uber_fare_pipeline.joblib` (produced by the training notebook), not
retrained on every request.
"""

import os
import joblib
from flask import Flask, render_template, request

from feature_engineering import (
    preprocess_and_predict,
    InputValidationError,
    CAR_CONDITION_ORDER,
    TRAFFIC_ORDER,
    WEATHER_OPTIONS,
)

MODEL_PATH = "uber_fare_pipeline.joblib"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained pipeline once at startup
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find '{MODEL_PATH}'. Run the training notebook first so it "
        f"saves the fitted pipeline (joblib.dump(tuned_pipeline, '{MODEL_PATH}')), "
        f"then place that file next to app.py."
    )

pipeline = joblib.load(MODEL_PATH)
print(f"Loaded model pipeline from '{MODEL_PATH}'.")


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        car_conditions=list(CAR_CONDITION_ORDER.keys()),
        traffic_conditions=list(TRAFFIC_ORDER.keys()),
        weather_options=WEATHER_OPTIONS,
        form_values={},
        error=None,
        prediction=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_values = request.form.to_dict()
    context = dict(
        car_conditions=list(CAR_CONDITION_ORDER.keys()),
        traffic_conditions=list(TRAFFIC_ORDER.keys()),
        weather_options=WEATHER_OPTIONS,
        form_values=form_values,
        error=None,
        prediction=None,
    )

    try:
        fare_usd, _ = preprocess_and_predict(form_values, pipeline)
        context["prediction"] = round(fare_usd, 2)
    except InputValidationError as e:
        context["error"] = str(e)
    except Exception as e:
        # Catch-all so a bad request never surfaces a raw stack trace to the user
        context["error"] = f"Something went wrong while processing your trip: {e}"

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
