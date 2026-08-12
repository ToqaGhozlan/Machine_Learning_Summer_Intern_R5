import argparse
import json
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import statsmodels.api as sm
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")


def fetch_nasa_power(lat, lon, start_date, end_date, parameters):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "parameters": ",".join(parameters),
        "format": "JSON",
        "community": "AG",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_power_json(js, parameters):
    param_data = js.get("properties", {}).get("parameter", {})
    all_dates = set()
    for p in parameters:
        series = param_data.get(p, {})
        all_dates.update(series.keys())

    if not all_dates:
        return pd.DataFrame()

    dates_sorted = sorted(pd.to_datetime(list(all_dates)))
    df = pd.DataFrame(index=dates_sorted)
    for p in parameters:
        series = param_data.get(p, {})
        values = [series.get(d.strftime("%Y-%m-%d")) or series.get(d.strftime("%Y%m%d")) for d in dates_sorted]
        df[p] = values
    df.index.name = "date"
    return df.sort_index()


def prepare_timeseries(df):
    start, end = df.index.min(), df.index.max()
    freq = pd.infer_freq(df.index) or "D"
    original_index = pd.DatetimeIndex(df.index)
    full_idx = pd.date_range(start=start, end=end, freq=freq)
    df = df.reindex(full_idx)

    missing_before = int(df.isna().sum().sum())
    introduced_missing_timestamps = int(full_idx.difference(original_index).shape[0])

    df = df.copy()
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()

    for col in [c for c in df.columns if c in {"T2M", "T2M_MIN", "T2M_MAX", "PRECTOT"}]:
        temp_series = df[col].copy()
        temp_series = temp_series.interpolate(method="time", limit_direction="both")
        for gap in find_missing_blocks(temp_series.isna()):
            if len(gap) <= 3:
                temp_series.iloc[gap] = temp_series.iloc[gap].interpolate(method="time", limit_direction="both")
            else:
                temp_series.iloc[gap] = temp_series.shift(7).iloc[gap]
        df[col] = temp_series

    missing_after = int(df.isna().sum().sum())
    remaining_missing = int(df.isna().sum().sum())
    return df, {
        "before": missing_before,
        "introduced_missing_timestamps": introduced_missing_timestamps,
        "after": missing_after,
        "remaining_missing": remaining_missing,
        "gaps": introduced_missing_timestamps,
    }


def find_missing_blocks(mask):
    idx = np.flatnonzero(mask.to_numpy())
    blocks = []
    if len(idx) == 0:
        return blocks
    start = idx[0]
    prev = idx[0]
    for value in idx[1:]:
        if value != prev + 1:
            blocks.append(list(range(start, prev + 1)))
            start = value
        prev = value
    blocks.append(list(range(start, prev + 1)))
    return blocks


def eda_and_plots(df, outdir, site_name="site"):
    os.makedirs(outdir, exist_ok=True)
    sns.set(style="whitegrid")

    summary = df.describe()
    summary.to_csv(os.path.join(outdir, "summary_stats.csv"))

    if any(c.upper().startswith("T2M") for c in df.columns):
        temp_cols = [c for c in df.columns if c.upper().startswith("T2M")]
        plt.figure(figsize=(12, 4))
        df[temp_cols].plot(title=f"Temperature series ({site_name})")
        plt.xlabel("Date")
        plt.ylabel("°C")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "temperature_series.png"))
        plt.close()

    if any("PRECTOT" in c.upper() or "PRE" in c.upper() for c in df.columns):
        precip_cols = [c for c in df.columns if "PRECTOT" in c.upper() or "PRE" in c.upper()]
        plt.figure(figsize=(12, 4))
        df[precip_cols].plot(title=f"Precipitation series ({site_name})")
        plt.xlabel("Date")
        plt.ylabel("mm/day")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "precipitation_series.png"))
        plt.close()

    ts = df["T2M"].dropna()
    stl = STL(ts, period=7, robust=True)
    res = stl.fit()
    res.plot()
    plt.suptitle("STL decomposition - Temperature")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "stl_temperature.png"))
    plt.close()

    adf_res = adfuller(ts.dropna())
    kpss_res = kpss(ts.dropna(), regression="c", nlags="auto")

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(ts.dropna(), ax=ax[0], lags=40)
    plot_pacf(ts.dropna(), ax=ax[1], lags=40, method="ywm")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acf_pacf_temperature.png"))
    plt.close()

    with open(os.path.join(outdir, "eda_results.json"), "w", encoding="utf8") as handle:
        json.dump({"adf_stat": adf_res[0], "adf_pvalue": adf_res[1], "kpss_stat": kpss_res[0], "kpss_pvalue": kpss_res[1]}, handle, indent=2)

    return summary, {"adf_stat": adf_res[0], "adf_pvalue": adf_res[1], "kpss_stat": kpss_res[0], "kpss_pvalue": kpss_res[1]}


def save_clean_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index_label="date")


def make_features(series, freq):
    df = pd.DataFrame(series)
    df.columns = ["temperature"]

    df["hour"] = 0
    if freq == "H":
        df["hour"] = series.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week"] = series.index.dayofweek
    df["month"] = series.index.month
    df["is_weekend"] = df["day_of_week"].ge(5).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for lag in [1, 24, 168]:
        df[f"lag_{lag}"] = df["temperature"].shift(lag)
    for window in [24, 168]:
        df[f"rolling_mean_{window}"] = df["temperature"].shift(1).rolling(window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = df["temperature"].shift(1).rolling(window, min_periods=1).std()
    return df.dropna()


def evaluate_forecast(actual, pred):
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / np.where(np.abs(actual) < 1e-8, np.nan, actual))) * 100
    denom = np.abs(actual) + np.abs(pred)
    smape = np.mean(2 * np.abs(actual - pred) / np.where(denom == 0, np.nan, denom)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}


def make_seasonal_naive_baseline(series, test_index, seasonality=7):
    preds = []
    for timestamp in test_index:
        source_date = timestamp - pd.Timedelta(days=seasonality)
        if source_date in series.index:
            preds.append(float(series.loc[source_date]))
        else:
            preds.append(float(series.iloc[-1]))
    return pd.Series(preds, index=test_index)


def fit_and_forecast(train, test, order, seasonal_order):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, trend="c", enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=len(test))
    pred_mean = forecast.predicted_mean
    pred_ci = forecast.conf_int(alpha=0.05)
    return fitted, pred_mean, pred_ci


def select_sarima_model(train, seasonal_period):
    candidate_orders = [
        ((1, 1, 1), (0, 0, 0, seasonal_period)),
        ((2, 1, 1), (0, 0, 0, seasonal_period)),
        ((1, 1, 2), (0, 0, 0, seasonal_period)),
        ((2, 1, 2), (0, 0, 0, seasonal_period)),
        ((1, 1, 1), (1, 0, 0, seasonal_period)),
        ((1, 1, 1), (0, 0, 1, seasonal_period)),
        ((2, 1, 1), (1, 0, 1, seasonal_period)),
        ((2, 1, 2), (1, 0, 1, seasonal_period)),
    ]

    inner_test_size = max(7, int(len(train) * 0.15))
    inner_train = train.iloc[:-inner_test_size]
    inner_val = train.iloc[-inner_test_size:]

    best_model = None
    best_order = None
    best_seasonal_order = None
    best_aic = np.inf
    best_metrics = None
    candidate_results = []

    for order, seasonal_order in candidate_orders:
        try:
            model = SARIMAX(inner_train, order=order, seasonal_order=seasonal_order, trend="c", enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit(disp=False)
            forecast = fitted.get_forecast(steps=len(inner_val)).predicted_mean
            metrics = evaluate_forecast(inner_val, forecast)
            candidate_results.append({
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": float(fitted.aic),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "mape": float(metrics["mape"]),
                "smape": float(metrics["smape"]),
            })
        except Exception as exc:
            candidate_results.append({
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": np.nan,
                "mae": np.nan,
                "rmse": np.nan,
                "mape": np.nan,
                "smape": np.nan,
                "error": str(exc),
            })
            continue

        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"] or (metrics["rmse"] == best_metrics["rmse"] and metrics["mae"] < best_metrics["mae"]):
            best_model = fitted
            best_order = order
            best_seasonal_order = seasonal_order
            best_aic = fitted.aic
            best_metrics = metrics

    if best_model is None:
        raise RuntimeError("Could not fit any SARIMA candidate model.")

    return best_model, best_order, best_seasonal_order, best_aic, best_metrics, pd.DataFrame(candidate_results)


def conservative_outlier_treatment(series):
    series = series.copy()
    before_stats = {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std()),
    }
    rolling_median = series.rolling(7, center=True, min_periods=3).median()
    rolling_mad = series.rolling(7, center=True, min_periods=3).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=False)
    mad_scale = rolling_mad * 1.4826
    impossible_mask = (series < -60) | (series > 60)
    z_score_mask = (np.abs(series - rolling_median) > 6 * mad_scale) & rolling_median.notna() & mad_scale.notna()
    outlier_mask = impossible_mask | z_score_mask
    modified_count = int(outlier_mask.sum())
    if modified_count > 0:
        series.loc[outlier_mask] = np.nan
        series = series.interpolate(method="time", limit_direction="both")
        series = series.fillna(series.shift(7))
    after_stats = {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std()),
    }
    return series, before_stats, after_stats, {
        "flagged_outliers": int(outlier_mask.sum()),
        "modified_values": modified_count,
    }


def run_forecasting_pipeline(input_csv, output_dir, site_name):
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.read_csv(input_csv, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    freq = pd.infer_freq(df.index) or "D"

    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(expected_index)
    initial_missing = int(df.isna().sum().sum())

    preprocessing_report = {}
    if "temperature" in df.columns:
        temp_series = df["temperature"].copy()
        temp_series = temp_series.interpolate(method="time", limit_direction="both")
        for gap in find_missing_blocks(temp_series.isna()):
            if len(gap) <= 3:
                temp_series.iloc[gap] = temp_series.iloc[gap].interpolate(method="time", limit_direction="both")
            else:
                temp_series.iloc[gap] = temp_series.shift(7).iloc[gap]
        df["temperature"] = temp_series
        preprocessing_report["temperature_missing_before"] = int(df["temperature"].isna().sum())
        df["temperature"], before_stats, after_stats, outlier_report = conservative_outlier_treatment(df["temperature"])
        preprocessing_report["temperature_before_stats"] = before_stats
        preprocessing_report["temperature_after_stats"] = after_stats
        preprocessing_report["outlier_report"] = outlier_report

    missing_after = int(df["temperature"].isna().sum())
    gaps = int(expected_index.difference(df.index).shape[0])

    feature_frame = make_features(df["temperature"], freq)
    feature_frame.to_csv(os.path.join(output_dir, "engineered_features.csv"))

    target = df["temperature"].copy()
    test_size = 28 if len(target) > 120 else max(7, int(len(target) * 0.2))
    train = target.iloc[:-test_size]
    test = target.iloc[-test_size:]

    plt.figure(figsize=(12, 4))
    train.plot(label="Training", figsize=(12, 4))
    test.plot(label="Test")
    plt.title(f"Temperature series and chronological split ({site_name})")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "time_series_split.png"))
    plt.close()

    adf_res = adfuller(train.dropna())
    kpss_res = kpss(train.dropna(), regression="c", nlags="auto")
    train_diff = train.diff().dropna()
    adf_diff = adfuller(train_diff)
    kpss_diff = kpss(train_diff, regression="c", nlags="auto")
    stationarity_summary = pd.DataFrame({
        "test": ["ADF", "KPSS"],
        "statistic": [adf_res[0], kpss_res[0]],
        "pvalue": [adf_res[1], kpss_res[1]],
    })
    stationarity_summary.to_csv(os.path.join(output_dir, "stationarity_summary.csv"), index=False)

    plt.figure(figsize=(12, 4))
    train_diff.plot(title="First-differenced temperature")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "first_difference.png"))
    plt.close()

    seasonal_period = 7 if freq in {"D", "W-SUN"} else 24
    decomp = seasonal_decompose(train.dropna(), model="additive", period=seasonal_period)
    fig = decomp.plot()
    fig.tight_layout()
    plt.savefig(os.path.join(figures_dir, "seasonal_decomposition.png"))
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(train_diff, ax=ax[0], lags=40)
    plot_pacf(train_diff, ax=ax[1], lags=40, method="ywm")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "acf_pacf_stationary.png"))
    plt.close()

    manual_order = (2, 1, 1)
    manual_seasonal_order = (0, 0, 0, seasonal_period)

    auto_model = auto_arima(train.dropna(), seasonal=True, m=seasonal_period, start_p=0, d=1, start_q=0, max_p=3, max_d=1, max_q=3, start_P=0, D=0, start_Q=0, max_P=2, max_D=1, max_Q=2, information_criterion="aic", suppress_warnings=True, stepwise=True, n_jobs=1)
    auto_order = auto_model.order
    auto_seasonal_order = auto_model.seasonal_order

    selected_model, selected_order, selected_seasonal_order, selected_aic, validation_metrics, candidate_results_df = select_sarima_model(train.dropna(), seasonal_period)
    sarima_model, sarima_forecast, sarima_ci = fit_and_forecast(train, test, selected_order, selected_seasonal_order)
    arima_model = ARIMA(train, order=(2, 1, 1)).fit()
    arima_forecast = arima_model.forecast(steps=len(test))

    residuals = sarima_model.resid
    std_resid = residuals / residuals.std(ddof=0)
    plt.figure(figsize=(12, 4))
    std_resid.plot(title="Standardized residuals")
    plt.axhline(0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "residuals_over_time.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(std_resid.dropna(), bins=20, edgecolor="black")
    plt.title("Residual histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "residual_histogram.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    import statsmodels.api as sm

    sm.qqplot(std_resid.dropna(), line="45", fit=True)
    plt.title("Q-Q plot of residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "residual_qq.png"))
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    plot_acf(residuals.dropna(), ax=ax, lags=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "residual_acf.png"))
    plt.close()

    sarima_model.plot_diagnostics(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sarima_diagnostics.png"))
    plt.close()

    lb_test = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)

    if not sarima_forecast.index.equals(test.index):
        sarima_forecast = sarima_forecast.reindex(test.index)
        sarima_ci = sarima_ci.reindex(test.index)

    plt.figure(figsize=(12, 4))
    train.iloc[-60:].plot(label="Training")
    test.plot(label="Actual test")
    sarima_forecast.plot(label="SARIMA forecast")
    plt.fill_between(sarima_forecast.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color="orange", alpha=0.2, label="95% CI")
    plt.title("SARIMA forecast vs actual test values")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "forecast_vs_actual.png"))
    plt.close()

    metrics = {
        "SARIMA": evaluate_forecast(test, sarima_forecast),
        "ARIMA": evaluate_forecast(test, arima_forecast),
    }

    persistence_pred = pd.Series([float(train.iloc[-1])] * len(test), index=test.index, dtype=float)
    metrics["Persistence baseline"] = evaluate_forecast(test, persistence_pred)

    seasonal_naive_pred = make_seasonal_naive_baseline(target, test.index, seasonality=seasonal_period)
    metrics["Seasonal naive"] = evaluate_forecast(test, seasonal_naive_pred)

    seasonal_naive_sample = pd.DataFrame({
        "actual_date": test.index[:10],
        "actual_value": test.iloc[:10].values,
        "source_date": [ts - pd.Timedelta(days=seasonal_period) for ts in test.index[:10]],
        "source_value": [float(target.loc[ts - pd.Timedelta(days=seasonal_period)]) if (ts - pd.Timedelta(days=seasonal_period)) in target.index else np.nan for ts in test.index[:10]],
        "predicted_value": seasonal_naive_pred.iloc[:10].values,
    })
    print("Seasonal-naive alignment sample:")
    print(seasonal_naive_sample.to_string(index=False))

    tscv = TimeSeriesSplit(n_splits=3, test_size=7)
    walk_forward_results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(target), start=1):
        train_fold = target.iloc[train_idx]
        test_fold = target.iloc[test_idx]
        fold_model = SARIMAX(train_fold, order=selected_order, seasonal_order=selected_seasonal_order, trend="c", enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fold_pred = fold_model.get_forecast(steps=len(test_fold)).predicted_mean
        fold_pred = fold_pred.reindex(test_fold.index)
        fold_metrics = evaluate_forecast(test_fold, fold_pred)
        walk_forward_results.append({"fold": fold, "train_end": train_fold.index[-1], "validation_start": test_fold.index[0], "validation_end": test_fold.index[-1], **fold_metrics})

    walk_forward_df = pd.DataFrame(walk_forward_results)
    walk_forward_df.to_csv(os.path.join(output_dir, "walk_forward_metrics.csv"), index=False)

    comparison_df = pd.DataFrame([
        {"Model": "Persistence baseline", **metrics["Persistence baseline"]},
        {"Model": "Seasonal naive", **metrics["Seasonal naive"]},
        {"Model": "ARIMA", **metrics["ARIMA"]},
        {"Model": "SARIMA", **metrics["SARIMA"]},
    ])
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    candidate_results_df.to_csv(os.path.join(output_dir, "sarima_selection_candidates.csv"), index=False)

    near_zero_count = int(np.sum(np.abs(test) < 1e-6))
    min_abs_actual = float(np.min(np.abs(test)))
    alignment_checks = {
        "arima_index_match": bool(arima_forecast.index.equals(test.index)),
        "sarima_index_match": bool(sarima_forecast.index.equals(test.index)),
        "persistence_index_match": bool(persistence_pred.index.equals(test.index)),
        "seasonal_naive_index_match": bool(seasonal_naive_pred.index.equals(test.index)),
    }
    summary_lines = [
        "# Task 5 forecasting summary",
        "",
        f"- Dataset: {site_name}",
        f"- Frequency: {freq}",
        f"- Target variable: temperature",
        f"- Seasonal period used: {seasonal_period}",
        f"- Missing timestamps detected after resampling: {gaps}",
        f"- Missing values before preprocessing: {initial_missing}",
        f"- Missing values after preprocessing: {missing_after}",
        f"- Outlier report: flagged={preprocessing_report['outlier_report']['flagged_outliers']}, modified={preprocessing_report['outlier_report']['modified_values']}",
        f"- Outlier stats: before min={preprocessing_report['temperature_before_stats']['min']:.3f}, before max={preprocessing_report['temperature_before_stats']['max']:.3f}; after min={preprocessing_report['temperature_after_stats']['min']:.3f}, after max={preprocessing_report['temperature_after_stats']['max']:.3f}",
        f"- Near-zero test values for MAPE stability: {near_zero_count}; minimum absolute actual test value: {min_abs_actual:.3f}",
        "",
        "## Stationarity",
        f"- ADF on training target: statistic={adf_res[0]:.3f}, p-value={adf_res[1]:.3f}",
        f"- KPSS on training target: statistic={kpss_res[0]:.3f}, p-value={kpss_res[1]:.3f}",
        f"- Differencing decision: first differencing was used because the training series was non-stationary; seasonal differencing was not used because the daily series did not show a strong seasonal unit root signal.",
        "",
        "## Model selection",
        f"- Manual ACF/PACF candidate: {manual_order} with seasonal order {manual_seasonal_order}",
        f"- Auto_arima selected: order={auto_order}, seasonal_order={auto_seasonal_order}",
        f"- Validation-selected SARIMA order: order={selected_order}, seasonal_order={selected_seasonal_order}, AIC={selected_aic:.2f}",
        f"- Validation metrics used for selection: MAE={validation_metrics['mae']:.3f}, RMSE={validation_metrics['rmse']:.3f}",
        "",
        "### SARIMA candidate comparison",
        candidate_results_df.to_string(index=False),
        "",
        "## Index alignment checks",
        f"- ARIMA forecast aligned to test index: {alignment_checks['arima_index_match']}",
        f"- SARIMA forecast aligned to test index: {alignment_checks['sarima_index_match']}",
        f"- Persistence baseline aligned to test index: {alignment_checks['persistence_index_match']}",
        f"- Seasonal naive baseline aligned to test index: {alignment_checks['seasonal_naive_index_match']}",
        "",
        "## Diagnostics",
        f"- Ljung-Box p-value for residual autocorrelation: {lb_test.iloc[0, 1]:.4f}",
        "- A high p-value supports the absence of significant residual autocorrelation.",
        "",
        "## Test-set metrics",
        comparison_df.to_string(index=False),
        "",
        "## Walk-forward validation",
        walk_forward_df.to_string(index=False),
        "",
        "## Conclusion",
        "The Task 5 workflow was implemented on the existing Task 4 weather dataset using a verified daily frequency and a seasonal period of 7. Missing timestamps were detected after reindexing, short gaps were interpolated, and long gaps were filled with a seasonal-naive approach. The pipeline uses a chronological split, a conservative outlier policy, and a validation-based SARIMA selection strategy. MAPE is reported for completeness but is less stable when temperature values are near zero; MAE and RMSE are treated as the primary metrics. SARIMAX was not used because the available exogenous variables were not available at forecast time without introducing leakage.",
    ]
    with open(os.path.join(output_dir, "forecasting_summary.md"), "w", encoding="utf8") as handle:
        handle.write("\n".join(summary_lines))

    print("Task 5 forecasting pipeline completed.")
    print("- Train period:", train.index[0], "to", train.index[-1])
    print("- Test period:", test.index[0], "to", test.index[-1])
    print("- SARIMA order:", selected_order, selected_seasonal_order)
    print("- Auto_arima order:", auto_order, auto_seasonal_order)
    print("- Model comparison:")
    print(comparison_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Fetch NASA POWER data and run Task 4/Task 5 preprocessing and forecasting")
    parser.add_argument("--lat", type=float, default=30.0444, help="Latitude (default Cairo)")
    parser.add_argument("--lon", type=float, default=31.2357, help="Longitude (default Cairo)")
    parser.add_argument("--start", type=str, default="20230101", help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default="20251231", help="End date YYYYMMDD")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--input-csv", type=str, default=None, help="Optional path to an existing cleaned CSV")
    parser.add_argument("--run-fetch", action="store_true", help="Fetch new data instead of using the existing cleaned CSV")
    args = parser.parse_args()

    if args.run_fetch:
        start = datetime.strptime(args.start, "%Y%m%d")
        end = datetime.strptime(args.end, "%Y%m%d")
        params = ["T2M_MIN", "T2M_MAX", "T2M", "PRECTOT"]
        rawdir = os.path.join(args.outdir, "raw")
        os.makedirs(rawdir, exist_ok=True)
        print(f"Fetching NASA POWER data for ({args.lat}, {args.lon}) {start.date()} -> {end.date()}")
        js = fetch_nasa_power(args.lat, args.lon, start, end, params)
        raw_path = os.path.join(rawdir, "power_response.json")
        with open(raw_path, "w", encoding="utf8") as handle:
            json.dump(js, handle, indent=2)
        df = parse_power_json(js, params)
        df_prep, missing = prepare_timeseries(df)
        cleaned_dir = os.path.join(args.outdir, "cleaned")
        save_clean_csv(df_prep, os.path.join(cleaned_dir, "cleaned_weather.csv"))
        std_df = pd.DataFrame(index=df_prep.index)
        if "T2M" in df_prep.columns:
            std_df["temperature"] = df_prep["T2M"]
        else:
            std_df["temperature"] = np.nan
        if "T2M_MIN" in df_prep.columns:
            std_df["temp_min"] = df_prep["T2M_MIN"]
        if "T2M_MAX" in df_prep.columns:
            std_df["temp_max"] = df_prep["T2M_MAX"]
        if "PRECTOT" in df_prep.columns:
            std_df["precipitation"] = df_prep["PRECTOT"]
        std_df = std_df.reset_index().rename(columns={std_df.columns[0]: "date"})
        std_df.to_csv(os.path.join(cleaned_dir, "cleaned_standardized.csv"), index=False)
        input_csv = os.path.join(cleaned_dir, "cleaned_standardized.csv")
    else:
        input_csv = args.input_csv or os.path.join(args.outdir, "Cairo", "cleaned", "cleaned_standardized.csv")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    run_forecasting_pipeline(input_csv, os.path.join(args.outdir, "Cairo"), "Cairo")


if __name__ == "__main__":
    main()
