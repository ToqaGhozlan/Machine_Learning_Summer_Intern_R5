"""
Loads the trained SARIMAX(2,1,2)+Fourier weather model exactly once, when this
module is first imported (Django imports it once per worker process on
startup), and exposes a small clean interface for the views to call.

Why a date-based model instead of tabular inputs (temperature/humidity/wind):
see README.md — Task 5's model is a time-series forecaster fit on Algiers'
daily temperature history plus Fourier terms encoding the annual seasonal
cycle. It doesn't take current-conditions readings as input; it takes a
*future date* and extrapolates from the learned trend + seasonal pattern.
This module reflects that: `predict_for_date()` takes a date, not a features
dict.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

MAX_FORECAST_HORIZON_DAYS = 180  # forecast uncertainty grows quickly beyond this


@dataclass
class PredictionResult:
    target_date: dt.date
    days_ahead: int
    predicted_temp_c: float
    ci_lower_c: float
    ci_upper_c: float


class WeatherModel:
    """Thin wrapper around the fitted SARIMAXResults + metadata, loaded once."""

    def __init__(self, model_path):
        artifact = joblib.load(model_path)
        self.fit = artifact["fitted_model"]
        self.last_train_date: dt.date = artifact["last_train_date"].date()
        self.fourier_order: int = artifact["fourier_order"]
        self.n_train_obs: int = artifact["n_train_obs"]
        self.temperature_history: pd.Series = artifact["temperature_history"]
        self.region: str = artifact["region"]
        self.model_version: str = artifact["model_version"]

    def _fourier_terms(self, dates: pd.DatetimeIndex, start_t: int, period: float = 365.25) -> pd.DataFrame:
        t = np.arange(start_t, start_t + len(dates))
        terms = {}
        for k in range(1, self.fourier_order + 1):
            terms[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
            terms[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
        return pd.DataFrame(terms, index=dates)

    def first_train_date(self) -> dt.date:
        return self.temperature_history.index.min().date()

    def min_valid_date(self) -> dt.date:
        return self.last_train_date + dt.timedelta(days=1)

    def max_valid_date(self) -> dt.date:
        return self.last_train_date + dt.timedelta(days=MAX_FORECAST_HORIZON_DAYS)

    def predict_for_date(self, target_date: dt.date) -> PredictionResult:
        """Forecast temperature for a single future date. Raises ValueError if the
        date is out of the model's valid, supported range."""
        if target_date <= self.last_train_date:
            raise ValueError(
                f"Date must be after {self.last_train_date.isoformat()} "
                f"(the model's last training date)."
            )
        days_ahead = (target_date - self.last_train_date).days
        if days_ahead > MAX_FORECAST_HORIZON_DAYS:
            raise ValueError(
                f"Date is too far ahead ({days_ahead} days). "
                f"This model supports forecasts up to {MAX_FORECAST_HORIZON_DAYS} days "
                f"past {self.last_train_date.isoformat()} — forecast uncertainty grows "
                f"quickly beyond that horizon."
            )

        future_dates = pd.date_range(self.min_valid_date(), periods=MAX_FORECAST_HORIZON_DAYS, freq="D")
        exog_future = self._fourier_terms(future_dates, start_t=self.n_train_obs)
        forecast = self.fit.get_forecast(steps=MAX_FORECAST_HORIZON_DAYS, exog=exog_future)

        pred_mean = forecast.predicted_mean
        ci = forecast.conf_int(alpha=0.05)
        target_ts = pd.Timestamp(target_date)

        return PredictionResult(
            target_date=target_date,
            days_ahead=days_ahead,
            predicted_temp_c=round(float(pred_mean.loc[target_ts]), 2),
            ci_lower_c=round(float(ci.loc[target_ts].iloc[0]), 2),
            ci_upper_c=round(float(ci.loc[target_ts].iloc[1]), 2),
        )

    def recent_history(self, n_days: int = 30) -> list[dict]:
        """Last n days of actual training data, for the bonus recent-inputs chart."""
        tail = self.temperature_history.tail(n_days)
        return [{"date": d.strftime("%Y-%m-%d"), "temp": round(float(v), 2)} for d, v in tail.items()]


# --- Module-level singleton: loaded once when Django imports this module ---
_model_instance: WeatherModel | None = None


def get_model() -> WeatherModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = WeatherModel(settings.WEATHER_MODEL_PATH)
    return _model_instance
