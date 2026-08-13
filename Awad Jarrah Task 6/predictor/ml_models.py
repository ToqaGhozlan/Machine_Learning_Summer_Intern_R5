"""
Loads the three pickled models produced in Task 5 (arma_manual, armax_auto,
fourier_auto) exactly once, and exposes a single `predict()` function the
Django view calls.

Model recap (see the Task 5 notebook for how each was fit):
- arma_manual : statsmodels ARIMA(1,0,4) results object. No exogenous
                covariates -> only needs a forecast horizon ("days ahead").
- armax_auto  : pmdarima auto_arima result fit WITH 7 meteorological
                covariates as exog. Needs one row of same-day covariates.
- fourier_auto: pmdarima auto_arima result fit WITH the same 7 covariates
                PLUS 3 pairs of annual Fourier terms (sin/cos, k=1..3,
                period=365.25). Needs the 7 covariates AND a forecast date
                (to compute the Fourier terms correctly).
"""

import pickle
import datetime as dt

import numpy as np
import pandas as pd
from django.conf import settings

EXOG_COLS = [
    "humidity",
    "tempmax",
    "dew",
    "tempmin",
    "precip",
    "cloudcover",
    "precipprob",
]

MODEL_CHOICES = [
    ("arma_manual", "Manual ARMA (no weather covariates)"),
    ("armax_auto", "Auto ARMAX (uses weather covariates)"),
    ("fourier_auto", "Auto ARMAX + Fourier (best model, uses weather covariates + date)"),
]

_MODELS = {}
_LOAD_ERRORS = {}


def load_all_models():
    """Unpickle each saved model once and cache it in _MODELS."""
    for key, _label in MODEL_CHOICES:
        if key in _MODELS or key in _LOAD_ERRORS:
            continue
        path = settings.SAVED_MODELS_DIR / f"{key}.pkl"
        try:
            with open(path, "rb") as f:
                _MODELS[key] = pickle.load(f)
        except FileNotFoundError:
            _LOAD_ERRORS[key] = (
                f"Model file not found at {path}. Copy your saved "
                f"'{key}.pkl' from Task 5 into the saved_models/ folder."
            )
        except Exception as exc:  # noqa: BLE001 - surface any unpickle error to the UI
            _LOAD_ERRORS[key] = f"Failed to load '{key}.pkl': {exc}"


def get_load_errors():
    """Any models that failed to load, so the view/template can warn the user."""
    load_all_models()
    return _LOAD_ERRORS


def _fourier_terms(forecast_date):
    """Recreate the same annual Fourier features used at training time.

    t is measured in days since TRAIN_START_DATE, matching how the notebook
    built t_train = np.arange(len(train)) starting the day training began.
    """
    train_start = dt.date.fromisoformat(settings.TRAIN_START_DATE)
    t = (forecast_date - train_start).days

    terms = {}
    for k in range(1, 4):
        terms[f"sin_year_{k}"] = np.sin(2 * np.pi * k * t / 365.25)
        terms[f"cos_year_{k}"] = np.cos(2 * np.pi * k * t / 365.25)
    return terms


def predict(model_choice, cleaned_data):
    """
    Run the chosen model and return a dict with the prediction and any
    extra context the template wants to display.

    `cleaned_data` is the Django form's cleaned_data dict.
    Raises ValueError for anything the view should treat as a user-facing
    error message (e.g. model unavailable).
    """
    load_all_models()

    if model_choice in _LOAD_ERRORS:
        raise ValueError(_LOAD_ERRORS[model_choice])
    if model_choice not in _MODELS:
        raise ValueError(f"Unknown model '{model_choice}'.")

    model = _MODELS[model_choice]

    if model_choice == "arma_manual":
        forecast_date = cleaned_data["forecast_date"]
        train_end = dt.date.fromisoformat(settings.TRAIN_END_DATE)
        steps = (forecast_date - train_end).days  # convert the picked date to a horizon

        forecast_res = model.get_forecast(steps=steps)
        predicted_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05)

        predicted_temp = float(predicted_mean.iloc[-1])
        lower = float(conf_int.iloc[-1, 0])
        upper = float(conf_int.iloc[-1, 1])

        return {
            "predicted_temp": predicted_temp,
            "lower": lower,
            "upper": upper,
            "detail": f"Forecast for {forecast_date.isoformat()} "
                      f"({steps} day(s) past the end of the training data, {settings.TRAIN_END_DATE}).",
        }

    # Both armax_auto and fourier_auto share the same exog-based prediction shape
    exog_row = {col: cleaned_data[col] for col in EXOG_COLS}

    if model_choice == "fourier_auto":
        forecast_date = cleaned_data["forecast_date"]
        exog_row.update(_fourier_terms(forecast_date))
        detail = f"Forecast for {forecast_date.isoformat()} using your weather inputs + seasonal terms."
    else:
        detail = "Forecast using your weather inputs (next time step)."

    X = pd.DataFrame([exog_row])

    preds, conf_int = model.predict(n_periods=1, X=X, return_conf_int=True)
    predicted_temp = float(np.asarray(preds)[0])
    lower = float(np.asarray(conf_int)[0, 0])
    upper = float(np.asarray(conf_int)[0, 1])

    return {
        "predicted_temp": predicted_temp,
        "lower": lower,
        "upper": upper,
        "detail": detail,
    }
