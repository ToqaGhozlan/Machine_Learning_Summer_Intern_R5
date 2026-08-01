import os
import json
from datetime import datetime
import argparse

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def fetch_nasa_power(lat, lon, start_date, end_date, parameters):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "parameters": ",".join(parameters),
        "format": "JSON",
        "community": "AG"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def parse_power_json(js, parameters):
    # NASA POWER daily point returns data under ['properties']['parameter'][PARAM]
    param_data = js.get("properties", {}).get("parameter", {})
    # Collect union of dates across all parameters
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
        # create series aligned to full index
        values = [series.get(d.strftime("%Y-%m-%d")) or series.get(d.strftime("%Y%m%d")) for d in dates_sorted]
        df[p] = values
    df.index.name = "date"
    return df.sort_index()


def prepare_timeseries(df):
    # Ensure daily frequency
    start, end = df.index.min(), df.index.max()
    full_idx = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(full_idx)

    # Report missing
    missing = df.isna().sum()

    # Strategy: interpolate temperatures (linear), for precipitation keep zeros where small gaps
    temps = [c for c in df.columns if c.upper().startswith("T2M")]
    precip_cols = [c for c in df.columns if "PRECTOT" in c.upper() or "PRE" in c.upper()]

    if temps:
        df[temps] = df[temps].interpolate(method="time", limit=7)
    if precip_cols:
        # fill short gaps with 0 if surrounded by zeros, otherwise leave NaN
        df[precip_cols] = df[precip_cols].fillna(0)

    return df, missing


def eda_and_plots(df, outdir, site_name="site"):
    os.makedirs(outdir, exist_ok=True)
    sns.set(style="whitegrid")

    # Summary statistics
    summary = df.describe()
    summary.to_csv(os.path.join(outdir, "summary_stats.csv"))

    # Time series plots
    plt.figure(figsize=(12, 5))
    if any(c.upper().startswith("T2M") for c in df.columns):
        temp_cols = [c for c in df.columns if c.upper().startswith("T2M")]
        df[temp_cols].plot(title=f"Temperature series ({site_name})")
        plt.xlabel("Date")
        plt.ylabel("°C")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "temperature_series.png"))
        plt.close()

    if any("PRECTOT" in c.upper() or "PRE" in c.upper() for c in df.columns):
        precip_cols = [c for c in df.columns if "PRECTOT" in c.upper() or "PRE" in c.upper()]
        plt.figure(figsize=(12, 4))
        df[precip_cols].plot(title=f"Precipitation series ({site_name})", kind="line")
        plt.xlabel("Date")
        plt.ylabel("mm/day")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "precipitation_series.png"))
        plt.close()

    # Rolling stats
    for col in df.columns:
        plt.figure(figsize=(12, 4))
        df[col].rolling(window=30, min_periods=15).mean().plot(label="30d RM")
        df[col].rolling(window=30, min_periods=15).std().plot(label="30d STD")
        plt.legend()
        plt.title(f"Rolling stats - {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"rolling_{col}.png"))
        plt.close()

    # Monthly boxplot (seasonality)
    df_month = df.copy()
    df_month["month"] = df_month.index.month
    for col in [c for c in df.columns if not c == "month"]:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x="month", y=col, data=df_month.reset_index())
        plt.title(f"Monthly distribution - {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"monthly_box_{col}.png"))
        plt.close()

    # Decomposition and stationarity
    results = {}
    if any(c.upper().startswith("T2M") for c in df.columns):
        # pick mean temp if available
        if "T2M" in df.columns:
            ts = df["T2M"].dropna()
        else:
            # try mean of min/max
            tcols = [c for c in df.columns if c.upper().startswith("T2M")]
            ts = df[tcols].mean(axis=1).dropna()

        # STL decomposition
        stl = STL(ts, period=365, robust=True)
        res = stl.fit()
        res.plot()
        plt.suptitle("STL decomposition - Temperature")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "stl_temperature.png"))
        plt.close()

        # ADF test
        adf_res = adfuller(ts.dropna())
        results['adf_stat'] = adf_res[0]
        results['adf_pvalue'] = adf_res[1]

        # ACF/PACF
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(ts.dropna(), ax=ax[0], lags=60)
        plot_pacf(ts.dropna(), ax=ax[1], lags=60, method='ywm')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "acf_pacf_temperature.png"))
        plt.close()

    # Correlation matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='vlag')
    plt.title('Correlation matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'correlation_matrix.png'))
    plt.close()

    # Save results
    with open(os.path.join(outdir, "eda_results.json"), "w") as f:
        json.dump({"adf": results}, f, indent=2)

    return summary, results


def save_clean_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index_label="date")


def main():
    parser = argparse.ArgumentParser(description="Fetch NASA POWER daily data and run time-series EDA")
    parser.add_argument("--lat", type=float, default=30.0444, help="Latitude (default Cairo)")
    parser.add_argument("--lon", type=float, default=31.2357, help="Longitude (default Cairo)")
    parser.add_argument("--start", type=str, default="20230101", help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default="20251231", help="End date YYYYMMDD")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y%m%d")
    end = datetime.strptime(args.end, "%Y%m%d")
    params = ["T2M_MIN", "T2M_MAX", "T2M", "PRECTOT"]

    rawdir = os.path.join(args.outdir, "raw")
    os.makedirs(rawdir, exist_ok=True)

    print(f"Fetching NASA POWER data for ({args.lat}, {args.lon}) {start.date()} -> {end.date()}")
    js = fetch_nasa_power(args.lat, args.lon, start, end, params)
    raw_path = os.path.join(rawdir, "power_response.json")
    with open(raw_path, "w", encoding="utf8") as f:
        json.dump(js, f, indent=2)

    print("Parsing JSON into DataFrame...")
    df = parse_power_json(js, params)

    print("Preparing time series (handling missing values and frequency)...")
    df_prep, missing = prepare_timeseries(df)

    # Save raw cleaned DataFrame (original parameter names)
    cleaned_dir = os.path.join(args.outdir, "cleaned")
    cleaned_path = os.path.join(cleaned_dir, "cleaned_weather.csv")
    save_clean_csv(df_prep, cleaned_path)

    # Also save a standardized CSV with required columns: date, temperature, precipitation (+extras)
    std_path = os.path.join(cleaned_dir, "cleaned_standardized.csv")
    os.makedirs(cleaned_dir, exist_ok=True)
    std_df = pd.DataFrame(index=df_prep.index)
    # Temperature (prefer T2M; otherwise mean of min/max)
    if "T2M" in df_prep.columns:
        std_df["temperature"] = df_prep["T2M"]
    else:
        tmin = df_prep.columns.intersection(["T2M_MIN", "T2M- MIN", "T2M_MIN "])
        tmax = df_prep.columns.intersection(["T2M_MAX", "T2M- MAX", "T2M_MAX "])
        if len(tmin) and len(tmax):
            std_df["temperature"] = df_prep[tmin[0]].add(df_prep[tmax[0]]).div(2)
        else:
            std_df["temperature"] = np.nan
    # Extras: min/max if present
    if "T2M_MIN" in df_prep.columns:
        std_df["temp_min"] = df_prep["T2M_MIN"]
    if "T2M_MAX" in df_prep.columns:
        std_df["temp_max"] = df_prep["T2M_MAX"]
    # Precipitation
    if "PRECTOT" in df_prep.columns:
        std_df["precipitation"] = df_prep["PRECTOT"]
    else:
        # try variants
        pre = [c for c in df_prep.columns if "PRE" in c.upper()]
        if pre:
            std_df["precipitation"] = df_prep[pre[0]]
        else:
            std_df["precipitation"] = np.nan

    # Reset index to have 'date' column
    std_df = std_df.reset_index()
    std_df.rename(columns={std_df.columns[0]: "date"}, inplace=True)
    std_df.to_csv(std_path, index=False)

    print("Running EDA and saving plots...")
    eda_out = os.path.join(args.outdir, "eda")
    summary, results = eda_and_plots(df_prep, eda_out, site_name=f"{args.lat},{args.lon}")

    # Write a short summary
    summary_txt = os.path.join(args.outdir, "summary.md")
    with open(summary_txt, "w", encoding="utf8") as f:
        f.write("# Weather EDA Summary\n\n")
        f.write(f"Region (lat,lon): {args.lat}, {args.lon}\n\n")
        f.write(f"Date range: {start.date()} to {end.date()}\n\n")
        f.write("## Data quality notes\n")
        f.write(str(missing.to_dict()))
        f.write("\n\n")
        f.write("## Stationarity (ADF)\n")
        if results.get('adf_pvalue') is not None:
            f.write(f"ADF statistic: {results['adf_stat']:.4f}, p-value: {results['adf_pvalue']:.4f}\n")
            if results['adf_pvalue'] < 0.05:
                f.write("Series is likely stationary (reject H0 at 5% level).\n")
            else:
                f.write("Series is likely non-stationary (fail to reject H0 at 5% level).\n")

    print("Done. Outputs written to:", args.outdir)


if __name__ == "__main__":
    main()
