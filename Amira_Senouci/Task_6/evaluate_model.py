"""
Evaluate the deployed SARIMAX model using a chronological holdout.

The final deployment model is fitted on all available history, so it cannot be
evaluated honestly against data it was trained on. This script recreates the
same deterministic cleaned series, holds out the final 180 days, fits the same
SARIMAX(2,1,2) + annual Fourier model on the earlier data, and compares it with
a simple persistence baseline (forecast = last observed training temperature).

Outputs:
    evaluation_results.json
This file is consumed by the Django dashboard.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from prepare_model import build_clean_dataframe, fourier_terms, ORDER, FOURIER_ORDER

warnings.filterwarnings("ignore")

HOLDOUT_DAYS = 180
OUTPUT_PATH = Path("evaluation_results.json")


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "mae": round(float(np.mean(np.abs(actual - predicted))), 3),
        "rmse": round(float(np.sqrt(np.mean((actual - predicted) ** 2))), 3),
        "mape": round(float(np.mean(np.abs((actual - predicted) / actual)) * 100), 2),
    }


def main():
    df = build_clean_dataframe()
    train = df.iloc[:-HOLDOUT_DAYS]
    test = df.iloc[-HOLDOUT_DAYS:]

    exog_train = fourier_terms(train.index, start_t=0, order=FOURIER_ORDER)
    exog_test = fourier_terms(test.index, start_t=len(train), order=FOURIER_ORDER)

    model = SARIMAX(
        train["temperature"],
        exog=exog_train,
        order=ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    prediction = fit.get_forecast(steps=HOLDOUT_DAYS, exog=exog_test).predicted_mean

    actual = test["temperature"].to_numpy()
    sarimax_pred = np.asarray(prediction, dtype=float)

    # Simple persistence baseline: tomorrow/next days = last observed train value.
    baseline_pred = np.repeat(float(train["temperature"].iloc[-1]), HOLDOUT_DAYS)

    result = {
        "model": "SARIMAX(2,1,2) + Fourier(order=3)",
        "baseline": "Persistence (last observed temperature)",
        "holdout_days": HOLDOUT_DAYS,
        "training_start": train.index.min().date().isoformat(),
        "training_end": train.index.max().date().isoformat(),
        "test_start": test.index.min().date().isoformat(),
        "test_end": test.index.max().date().isoformat(),
        "metrics": {
            "sarimax": metrics(actual, sarimax_pred),
            "baseline": metrics(actual, baseline_pred),
        },
        "series": [
            {
                "date": date.strftime("%Y-%m-%d"),
                "actual": round(float(a), 2),
                "predicted": round(float(p), 2),
            }
            for date, a, p in zip(test.index, actual, sarimax_pred)
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Model evaluation completed.")
    print(f"Holdout: {result['test_start']} -> {result['test_end']} ({HOLDOUT_DAYS} days)")
    print("SARIMAX:", result["metrics"]["sarimax"])
    print("Baseline:", result["metrics"]["baseline"])
    print(f"Saved: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
