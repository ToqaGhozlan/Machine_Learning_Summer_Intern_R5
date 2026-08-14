"""
Loads the trained SARIMAX (order=(2,1,1)) + Fourier exogenous model once per process
and produces a one-day-ahead temperature forecast from user-supplied weather inputs.

The forecast is always for the first day after the training data ends
(model.nobs steps in), since that's the only point where "t" is unambiguous
without asking the user for a date.
"""
import threading
from pathlib import Path
from datetime import date, timedelta
from xml.parsers.expat import model

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

MODEL_PATH = Path(settings.BASE_DIR) / "models" / "sarima_fourier_model.pkl"

WEATHER_FEATURES = [
    "max_temperature",
    "min_temperature",
    "precipitation",
    "humidity",
    "wind_speed",
    "solar_radiation",
]

FOURIER_K = 3
ANNUAL_PERIOD = 365.25
TRAIN_START = date(2021, 1, 1)  # first date in the full dataset used to index Fourier terms

_model = None
_lock = threading.Lock()


def get_model():
    #Load the trained model once and reuse it across requests (thread-safe)
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = joblib.load(MODEL_PATH)
    return _model


def _fourier_terms(t: int) -> dict:
    terms = {}
    for k in range(1, FOURIER_K + 1):
        terms[f"sin_{k}"] = np.sin(2 * np.pi * k * t / ANNUAL_PERIOD)
        terms[f"cos_{k}"] = np.cos(2 * np.pi * k * t / ANNUAL_PERIOD)
    return terms

# helps to predict the next day temperature based on the user input weather values
def predict_next_day(weather_values: dict) -> dict:
    model = get_model()

    t_next = model.nobs  # first out-of-sample step, continuing the training day-index
    forecast_date = TRAIN_START + timedelta(days=t_next)

    row = {feat: weather_values[feat] for feat in WEATHER_FEATURES}
    row.update(_fourier_terms(t_next))
    exog_row = pd.DataFrame([row], columns=model.model.exog_names)

    forecast = model.get_forecast(steps=1, exog=exog_row)
    mean = float(forecast.predicted_mean.iloc[0])
    ci = forecast.conf_int(alpha=0.05).iloc[0]

    return {
        "forecast_date": forecast_date,
        "predicted_temperature": mean,
        "lower_bound": float(ci.iloc[0]),
        "upper_bound": float(ci.iloc[1]),
    }
