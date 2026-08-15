from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "weather_model.pkl"
)


FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
]


# Load model once
MODEL = joblib.load(MODEL_PATH)


def predict_weather(values):

    data = np.array(
        [[
            values["temperature"],
            values["humidity"],
            values["wind_speed"],
            values["pressure"],
        ]],
        dtype=float
    )


    prediction = MODEL.predict(data)[0]


    return float(prediction)