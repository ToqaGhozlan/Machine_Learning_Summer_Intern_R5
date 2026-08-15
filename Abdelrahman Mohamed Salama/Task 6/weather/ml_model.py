"""this module provides functionality to load a pre-trained SARIMA model and
    make predictions for a chosen future date, up to 30 days after the
    training data ends. Since the model needs weather values (temperature,
    humidity, wind speed, etc.) as exogenous inputs for every day it
    forecasts, and the user now only supplies a date, those values are
    filled in from seasonal climate normals for Alexandria (see
    climatology.py) rather than typed in by hand.
    It includes thread-safe loading of the model, generation of Fourier
    terms for seasonal adjustment, and prediction with confidence
    intervals."""
import threading
from pathlib import Path
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

from .climatology import estimate_weather_for_date

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
TEST_END = date(2025, 12, 31)   # last date covered by train+test data (Task 5 evaluation)
MAX_HORIZON_DAYS = 30  # how many days past TEST_END the app allows forecasts for

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


def get_forecast_window() -> tuple[date, date]:
    """The range of dates the app will offer to forecast: the first day
    right after ALL known data ends (training + the held-out test set,
    through TEST_END), through MAX_HORIZON_DAYS after that. This is
    deliberately later than the model's own out-of-sample start
    (TRAIN_START + model.nobs days), since that earlier stretch falls
    inside the test period and was already observed."""
    first_date = TEST_END + timedelta(days=1)
    last_date = first_date + timedelta(days=MAX_HORIZON_DAYS - 1)
    return first_date, last_date


# helps to predict the temperature for a chosen future date, using
# climatology-estimated weather inputs for every day between the end of
# training and the target date
def predict_for_date(target_date: date) -> dict:
    model = get_model()

    window_first, window_last = get_forecast_window()
    if target_date < window_first or target_date > window_last:
        raise ValueError(
            f"Date must be between {window_first.isoformat()} and {window_last.isoformat()}."
        )

    # The SARIMAX object itself only knows it was fit on `nobs` days from
    # TRAIN_START - it has no concept of your train/test split. To reach a
    # date past TEST_END we still have to walk it forward step-by-step from
    # its own out-of-sample start, filling every intermediate day (including
    # the held-out test period) with climatology since we don't feed it the
    # real observed test-period weather here.
    t_start = model.nobs
    model_forecast_start = TRAIN_START + timedelta(days=t_start)
    steps = (target_date - model_forecast_start).days + 1

    rows = []
    for i in range(steps):
        t = t_start + i
        day = model_forecast_start + timedelta(days=i)
        row = estimate_weather_for_date(day)
        row.update(_fourier_terms(t))
        rows.append(row)

    exog = pd.DataFrame(rows, columns=model.model.exog_names)

    forecast = model.get_forecast(steps=steps, exog=exog)
    mean = float(forecast.predicted_mean.iloc[-1])
    ci = forecast.conf_int(alpha=0.05).iloc[-1]

    return {
        "forecast_date": target_date,
        "predicted_temperature": mean,
        "lower_bound": float(ci.iloc[0]),
        "upper_bound": float(ci.iloc[1]),
        "estimated_inputs": {feat: rows[-1][feat] for feat in WEATHER_FEATURES},
    }
