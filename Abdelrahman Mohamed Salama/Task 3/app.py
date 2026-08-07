# flask entry point ,this file only wires things together
import os
import logging

from flask import Flask, render_template, request, jsonify
import joblib

from validator import validate_trip_details, ValidationError
from predict import predict_fare

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fare_pipeline.pkl"),
)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


model = None
model_load_error = None
try:
    model = joblib.load(MODEL_PATH)  #Load the model once at startup--------------------------------------
    app.logger.info(f"Loaded model from {MODEL_PATH}")
except Exception as exc:
    model_load_error = str(exc)
    app.logger.warning(
        f"Could not load model from '{MODEL_PATH}': {exc}. "
        f"Copy your saved fare_pipeline.pkl"
        f"(or set the MODEL_PATH env var) before predicting."
    )

#index.html here------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

#js sends data to predict function-------------------------
@app.route("/predict", methods=["POST"])
def predict():
    
    if model is None:
        return jsonify({
            "error": f"Model not loaded on the server ({model_load_error}). "
        }), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Expected a JSON request body."}), 400

    try:
        trip = validate_trip_details(payload) # validates trip attributes-------------------------------------------
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

    try:
        result = predict_fare(model, trip)# predict the fare amount ----------------------
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Something went wrong while computing the fare. "
                     "Please check your inputs and try again."
        }), 500

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False,  host="0.0.0.0", port=9000)
