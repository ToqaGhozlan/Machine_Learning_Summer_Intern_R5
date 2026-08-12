# Time Series Preprocessing & Forecasting — Algiers Weather

**ML Internship Program @ Cellula Technologies** — Task 5: continues from Task 4's Algiers
weather series, extended to **5 years of daily data (2021–2025)**. Full forecasting
pipeline: time-aware preprocessing, ADF/KPSS stationarity testing, ARIMA/SARIMA modeling,
rigorous evaluation (naive baselines + walk-forward validation), and a bonus LSTM/RNN
comparison.

## Repository structure

```
01-Time_Series_Forecasting.ipynb   Full notebook: preprocessing → SARIMA → evaluation → LSTM
raw_nasa_power_response_5y.json    Raw API response for the 5-year pull
assets/                            Exported figures from the notebook
```

## Key result

| Model | MAE (°C) | RMSE (°C) | MAPE (%) |
|---|---|---|---|
| **SARIMAX(2,1,2) + Fourier terms** | **1.05** | **1.32** | **7.17** |
| LSTM | 1.20 | 1.50 | 8.27 |
| Simple RNN | 1.18 | 1.44 | 8.13 |
| Naive (persistence) | 1.49 | 1.90 | 10.11 |
| Seasonal-naive (t-365) | 1.63 | 1.98 | 11.07 |

All models beat both naive baselines; the classical seasonal model edges out both neural
approaches on this dataset — a plausible result for a smooth, strongly periodic ~1,800-point
daily series, where a well-specified harmonic-regression SARIMA captures the dominant annual
cycle almost by construction, while LSTMs typically need more data/tuning to reliably win on
series like this.

## A necessary adaptation from the assignment template

The assignment's SARIMA example (`seasonal_order=(...,24)`) targets **hourly** data with a
**daily** cycle (m=24) — small and computationally trivial. This dataset is **daily** with an
**annual** cycle (m=365): a literal `SARIMAX(seasonal_order=(...,365))` is not practically
fittable (the state-space model's size scales with the seasonal period). We instead use
**dynamic harmonic regression** — Fourier-term exogenous regressors representing the annual
cycle, combined with a non-seasonal ARIMA for short-range autocorrelation — the standard,
literature-backed substitute for long seasonal periods (Hyndman & Athanasopoulos). Explained
in full in the notebook's introduction and Section 4.

## Data provenance

Same NASA POWER `temporal/daily/point` client as Task 4. The real API client is called first;
this sandbox's network is restricted, so it falls back to a seeded synthetic generator
matching NASA POWER's schema (documented inline). Re-running with normal internet access
pulls live data with no code changes.

## Reproducing

```bash
pip install -r requirements.txt
jupyter notebook 01-Time_Series_Forecasting.ipynb
```

## Tools

Python · pandas · NumPy · Matplotlib · SciPy · statsmodels · pmdarima · scikit-learn ·
TensorFlow/Keras · requests
