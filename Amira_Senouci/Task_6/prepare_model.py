"""
Regenerates the Task 4/5 Algiers weather series and refits the final
SARIMAX(2,1,2) + annual-Fourier model on the FULL dataset (train+test),
since a deployed model should use all available history, not hold back
a test set. Saves the fitted model + metadata needed by the Django app.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
OUTPUT_DIR = Path(".")

LATITUDE, LONGITUDE = 36.75, 3.06
START_DATE, END_DATE = "20210101", "20251231"
PARAMETERS = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "WS2M", "CLOUD_AMT"]
FOURIER_ORDER = 3
ORDER = (2, 1, 2)


def simulate_nasa_power_response(lat, lon, start, end, parameters, seed=RANDOM_SEED):
    """Same seeded synthetic generator used in Task 4/5 (see those notebooks for the
    full rationale) — reproduced here so this script is self-contained."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(pd.to_datetime(start, format="%Y%m%d"),
                           pd.to_datetime(end, format="%Y%m%d"), freq="D")
    n = len(dates)
    day_of_year = dates.dayofyear.values
    years_elapsed = ((dates - dates[0]).days.values) / 365.25

    seasonal = 8.5 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
    trend = 0.30 * years_elapsed
    noise = rng.normal(0, 1.3, n)
    t2m = 18.3 + seasonal + trend + noise
    t2m_max = t2m + rng.uniform(3.5, 6.5, n)
    t2m_min = t2m - rng.uniform(3.0, 5.5, n)

    month = dates.month.values
    rain_prob = np.where(np.isin(month, [11, 12, 1, 2, 3]), 0.38,
                 np.where(np.isin(month, [4, 5, 10]), 0.22,
                 np.where(np.isin(month, [6, 9]), 0.08, 0.02)))
    rain_occurs = rng.random(n) < rain_prob
    precip = np.where(rain_occurs, rng.gamma(shape=1.6, scale=6.0, size=n), 0.0)
    rh2m = np.clip(68 - 0.9 * seasonal + rng.normal(0, 5, n) + 6 * rain_occurs, 35, 97)
    ws2m = np.clip(3.4 + 0.6 * np.cos(2 * np.pi * day_of_year / 365.25) + rng.normal(0, 0.8, n), 0.3, None)
    cloud_amt = np.clip(35 + 45 * rain_occurs + rng.normal(0, 12, n), 0, 100)

    data = {"T2M": t2m, "T2M_MAX": t2m_max, "T2M_MIN": t2m_min,
            "PRECTOTCORR": precip, "RH2M": rh2m, "WS2M": ws2m, "CLOUD_AMT": cloud_amt}

    date_keys = [d.strftime("%Y%m%d") for d in dates]
    n_missing = max(int(n * 0.008), 5)
    for i in rng.choice(n, size=n_missing, replace=False):
        data["T2M"][i] = -999.0
    for i in rng.choice(n, size=n_missing + 4, replace=False):
        data["PRECTOTCORR"][i] = -999.0
    for i in rng.choice(n, size=4, replace=False):
        data["T2M"][i] = rng.uniform(48, 57)
    for i in rng.choice(n, size=4, replace=False):
        data["PRECTOTCORR"][i] = -rng.uniform(3, 15)

    param_dict = {p: {date_keys[i]: round(float(data[p][i]), 2) for i in range(n)} for p in parameters}
    raw = {"properties": {"parameter": param_dict}}
    n_dupes = max(int(n * 0.002), 3)
    n_gaps = max(int(n * 0.003), 5)
    raw["_duplicate_dates"] = [date_keys[i] for i in rng.choice(n, size=n_dupes, replace=False)]
    raw["_drop_dates"] = [date_keys[i] for i in rng.choice(n, size=n_gaps, replace=False)]
    return raw


def build_clean_dataframe():
    raw_response = simulate_nasa_power_response(LATITUDE, LONGITUDE, START_DATE, END_DATE, PARAMETERS)
    param_data = raw_response["properties"]["parameter"]
    df_raw = pd.DataFrame({"temperature": pd.Series(param_data["T2M"])})
    df_raw.index = pd.to_datetime(df_raw.index, format="%Y%m%d")
    df_raw = df_raw.sort_index()

    dup_rows = df_raw.loc[pd.to_datetime(raw_response["_duplicate_dates"], format="%Y%m%d")]
    df_raw = pd.concat([df_raw, dup_rows]).sort_index()
    drop_idx = pd.to_datetime(raw_response["_drop_dates"], format="%Y%m%d")
    df_raw = df_raw.drop(index=drop_idx)

    df = df_raw[~df_raw.index.duplicated(keep="first")].copy()
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "date"
    df.index.freq = "D"

    df = df.replace(-999.0, np.nan)
    df["temperature"] = df["temperature"].interpolate(method="time", limit_direction="both")

    # Cap outliers via rolling z-score (same approach as Task 5)
    roll_mean = df["temperature"].rolling(30, center=True, min_periods=10).mean()
    roll_std = df["temperature"].rolling(30, center=True, min_periods=10).std()
    z = (df["temperature"] - roll_mean) / roll_std
    df.loc[z > 3.5, "temperature"] = roll_mean[z > 3.5] + 3.5 * roll_std[z > 3.5]
    df.loc[z < -3.5, "temperature"] = roll_mean[z < -3.5] - 3.5 * roll_std[z < -3.5]

    return df


def fourier_terms(index, start_t, period=365.25, order=FOURIER_ORDER):
    t = np.arange(start_t, start_t + len(index))
    terms = {}
    for k in range(1, order + 1):
        terms[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        terms[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(terms, index=index)


if __name__ == "__main__":
    df = build_clean_dataframe()
    print(f"Full cleaned series: {df.index.min().date()} -> {df.index.max().date()}  ({len(df)} rows)")

    exog = fourier_terms(df.index, start_t=0)
    model = SARIMAX(df["temperature"], exog=exog, order=ORDER,
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    print(fit.summary())

    artifact = {
        "fitted_model": fit,
        "last_train_date": df.index.max(),
        "n_train_obs": len(df),
        "fourier_order": FOURIER_ORDER,
        "sarimax_order": ORDER,
        "temperature_history": df["temperature"],   # kept small: 1826 floats, needed for the bonus chart
        "region": "Algiers, Algeria",
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "model_version": "sarimax_2_1_2_fourier3_v1",
    }
    joblib.dump(artifact, OUTPUT_DIR / "weather_model.pkl")
    print(f"\nSaved weather_model.pkl  (last training date: {df.index.max().date()})")

    # Quick sanity check: forecast 10 days ahead
    exog_future = fourier_terms(pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=10, freq="D"),
                                 start_t=len(df))
    fc = fit.get_forecast(steps=10, exog=exog_future)
    print("\nSanity-check 10-day forecast:")
    print(fc.predicted_mean.round(2))
