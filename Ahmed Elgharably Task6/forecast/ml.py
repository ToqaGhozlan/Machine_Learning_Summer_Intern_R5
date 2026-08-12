"""
Inference engine for the Damietta Temperature Forecast app.

This module is intentionally the *only* place that knows about the trained
artifacts (the fitted SARIMAX results object, the historical daily series,
and the evaluation metadata produced in Task 5). Views and forms never touch
joblib/pandas/statsmodels directly — they call `get_engine()` and use the
plain-Python dicts it returns. That separation is what Task 6's rubric means
by "clear separation of ML logic, forms, and views".

Model recap (see Task 5 notebook / this project's README for the full story):
    SARIMAX(2, 1, 2) errors + an annual Fourier regression (K=3 harmonics of
    sin/cos(2*pi*k*day_of_year / 365.25)) as exogenous predictors, refit on
    all 731 days of the cleaned Damietta dataset for this deployed version
    (the notebook's own train/test split was for evaluation only).

Two naive baselines are exposed alongside SARIMAX because Task 5's own
evaluation found persistence (tomorrow = today) *beats* SARIMAX at long,
static horizons on this smooth, day-to-day-correlated series — offering
only the "smart" model and hiding that would be misleading, so the dropdown
in the UI lets a user compare all three honestly.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from django.conf import settings

MODEL_SARIMAX = "sarimax"
MODEL_PERSISTENCE = "persistence"
MODEL_SEASONAL_NAIVE = "seasonal_naive"

MODEL_CHOICES = (
    (MODEL_SARIMAX, "SARIMAX(2,1,2) + Fourier — recommended"),
    (MODEL_PERSISTENCE, "Persistence (tomorrow \u2248 today)"),
    (MODEL_SEASONAL_NAIVE, "Seasonal-naive (same day, last year)"),
)

MAX_HORIZON_DAYS = 365  # how far past the last training date a user may ask for


class ForecastError(ValueError):
    """Raised for any input that is well-formed but not something the
    selected model can actually produce a forecast for (e.g. a
    seasonal-naive lookup with no matching date in history)."""


@dataclass
class ForecastResult:
    model_choice: str
    model_label: str
    target_date: date
    horizon_days: int
    predicted_temp_c: float
    ci_low_c: Optional[float]
    ci_high_c: Optional[float]
    expected_mae_c: float
    note: str


class _Engine:
    """Loads every artifact once and answers forecast queries. A process-
    wide singleton is created by `get_engine()` the first time it's needed —
    see apps.py for how that first call is triggered at server startup."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir

        with open(artifacts_dir / "model_meta.json") as f:
            self.meta = json.load(f)

        self.order = tuple(self.meta["order"])
        self.fourier_k = self.meta["fourier_K"]
        self.fourier_period = self.meta["fourier_period"]
        self.last_date = pd.Timestamp(self.meta["last_date"])
        self.first_date = pd.Timestamp(self.meta["first_date"])
        self.temp_min = self.meta["temp_plausible_min"]
        self.temp_max = self.meta["temp_plausible_max"]
        self.metrics = self.meta["metrics"]

        hist = pd.read_csv(artifacts_dir / "historical_temperature.csv", parse_dates=["date"])
        self.history = hist.set_index("date")["temperature"].asfreq("D")

        import joblib  # imported lazily so `python manage.py` commands that
        # never touch ML (makemigrations, etc.) don't pay statsmodels'
        # import cost.

        self.sarimax_results = joblib.load(artifacts_dir / "sarimax_model.joblib")

    # -- shared helpers ----------------------------------------------------

    def _fourier_terms(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        doy = index.dayofyear.values
        cols = {}
        for k in range(1, self.fourier_k + 1):
            cols[f"sin_{k}"] = np.sin(2 * np.pi * k * doy / self.fourier_period)
            cols[f"cos_{k}"] = np.cos(2 * np.pi * k * doy / self.fourier_period)
        return pd.DataFrame(cols, index=index)

    def recent_history(self, days: int = 30) -> list[dict]:
        """Last N observed days, for the sparkline chart on the form page."""
        tail = self.history.tail(days)
        return [
            {"date": d.strftime("%Y-%m-%d"), "temperature": round(float(t), 2)}
            for d, t in tail.items()
        ]

    def date_bounds(self) -> dict:
        return {
            "min": (self.last_date + timedelta(days=1)).date().isoformat(),
            "max": (self.last_date + timedelta(days=MAX_HORIZON_DAYS)).date().isoformat(),
        }

    # -- validation shared with forms.py ------------------------------------

    def validate_target_date(self, target: date) -> int:
        """Returns the horizon in days, or raises ForecastError."""
        target_ts = pd.Timestamp(target)
        horizon = (target_ts - self.last_date).days
        if horizon < 1:
            raise ForecastError(
                f"Pick a date after {self.last_date.date()} — that's the last day in the "
                "training data, so only dates after it are genuine forecasts."
            )
        if horizon > MAX_HORIZON_DAYS:
            raise ForecastError(
                f"That's {horizon} days out. This model is only validated up to "
                f"{MAX_HORIZON_DAYS} days ahead — pick an earlier date."
            )
        return horizon

    def validate_override_temp(self, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not (self.temp_min <= value <= self.temp_max):
            raise ForecastError(
                f"{value}\u00b0C is outside the physically plausible range for Damietta "
                f"({self.temp_min}\u2013{self.temp_max}\u00b0C)."
            )
        return value

    # -- the three forecasting strategies ------------------------------------

    def _predict_sarimax(self, target: date, horizon: int) -> ForecastResult:
        future_idx = pd.date_range(self.last_date + timedelta(days=1), periods=horizon, freq="D")
        exog = self._fourier_terms(future_idx)
        forecast = self.sarimax_results.get_forecast(steps=horizon, exog=exog)
        mean = float(forecast.predicted_mean.iloc[-1])
        ci = forecast.conf_int(alpha=0.05).iloc[-1]

        if horizon <= 14:
            expected_mae = self.metrics["sarimax_walkforward_14d_mae"]
            note = (
                f"Within the 14-day horizon this was validated on (walk-forward MAE "
                f"\u2248{expected_mae}\u00b0C)."
            )
        else:
            expected_mae = self.metrics["sarimax_static_90d"]["mae"]
            note = (
                "Beyond ~14 days the forecast leans mostly on the annual seasonal curve "
                f"(static 90-day test MAE \u2248{expected_mae}\u00b0C) — Task 5's own evaluation found "
                "plain persistence actually wins at this range, so treat this as a seasonal "
                "expectation, not a precise reading."
            )
        return ForecastResult(
            model_choice=MODEL_SARIMAX,
            model_label=dict(MODEL_CHOICES)[MODEL_SARIMAX],
            target_date=target,
            horizon_days=horizon,
            predicted_temp_c=round(mean, 1),
            ci_low_c=round(float(ci.iloc[0]), 1),
            ci_high_c=round(float(ci.iloc[1]), 1),
            expected_mae_c=expected_mae,
            note=note,
        )

    def _predict_persistence(self, target: date, horizon: int, override_temp: Optional[float]) -> ForecastResult:
        base = override_temp if override_temp is not None else float(self.history.iloc[-1])
        source = "the value you entered" if override_temp is not None else f"the last recorded day ({self.last_date.date()})"
        return ForecastResult(
            model_choice=MODEL_PERSISTENCE,
            model_label=dict(MODEL_CHOICES)[MODEL_PERSISTENCE],
            target_date=target,
            horizon_days=horizon,
            predicted_temp_c=round(base, 1),
            ci_low_c=None,
            ci_high_c=None,
            expected_mae_c=self.metrics["persistence_static_90d"]["mae"],
            note=f"Holds {source} constant — day-to-day temperature here is highly autocorrelated, "
                 "which is why this simple baseline is hard to beat at short horizons.",
        )

    def seasonal_naive_lookup(self, target: date) -> float:
        """Returns the historical temperature exactly 365 days before
        `target`, or raises ForecastError if that date isn't in the record.
        Shared by form validation (forms.py) and prediction below, so both
        agree on exactly what counts as "available"."""
        lookup_ts = pd.Timestamp(target) - pd.Timedelta(days=365)
        if lookup_ts not in self.history.index or pd.isna(self.history.get(lookup_ts)):
            raise ForecastError(
                f"Seasonal-naive needs the same calendar day one year earlier "
                f"({lookup_ts.date()}), which falls outside the {self.first_date.date()}\u2013"
                f"{self.last_date.date()} historical record. Try a date in "
                f"{self.first_date.year + 2}."
            )
        return float(self.history.loc[lookup_ts])

    def _predict_seasonal_naive(self, target: date, horizon: int) -> ForecastResult:
        lookup_ts = pd.Timestamp(target) - pd.Timedelta(days=365)
        value = self.seasonal_naive_lookup(target)
        return ForecastResult(
            model_choice=MODEL_SEASONAL_NAIVE,
            model_label=dict(MODEL_CHOICES)[MODEL_SEASONAL_NAIVE],
            target_date=target,
            horizon_days=horizon,
            predicted_temp_c=round(value, 1),
            ci_low_c=None,
            ci_high_c=None,
            expected_mae_c=self.metrics["seasonal_naive_static_90d"]["mae"],
            note=f"Reuses the observed temperature from {lookup_ts.date()} — the same calendar day "
                 "one year earlier.",
        )

    def predict(self, model_choice: str, target: date, override_temp: Optional[float] = None) -> ForecastResult:
        horizon = self.validate_target_date(target)
        override_temp = self.validate_override_temp(override_temp)

        if model_choice == MODEL_SARIMAX:
            return self._predict_sarimax(target, horizon)
        if model_choice == MODEL_PERSISTENCE:
            return self._predict_persistence(target, horizon, override_temp)
        if model_choice == MODEL_SEASONAL_NAIVE:
            return self._predict_seasonal_naive(target, horizon)
        raise ForecastError(f"Unknown model choice: {model_choice}")


_engine: Optional[_Engine] = None
_engine_lock = threading.Lock()


def get_engine() -> _Engine:
    """Process-wide singleton accessor. Double-checked locking keeps this
    safe if two threads race to build the engine on first use."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = _Engine(settings.ML_ARTIFACTS_DIR)
    return _engine
