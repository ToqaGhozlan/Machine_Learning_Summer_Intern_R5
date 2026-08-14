# Egypt Weather Forecasting (T2M Time Series)

A time series forecasting project that predicts daily average temperature (T2M) across Egypt using 10 years of NASA POWER weather data. The final model is an LSTM network, trained and exported for deployment through a Django web application.

## Project Overview

This project covers the full pipeline from raw data collection to a deployable forecasting model:

1. Collect 10 years (2015-2024) of daily meteorological data for all 27 Egyptian governorates from the NASA POWER API.
2. Clean, validate, and explore the data.
3. Build an Egypt-wide daily average temperature series.
4. Test several forecasting approaches and compare their accuracy.
5. Save the best-performing model for use in a Django application.

## Data Source

- **API**: [NASA POWER](https://power.larc.nasa.gov/) daily point API
- **Coverage**: 27 Egyptian governorates, using their approximate latitude/longitude centers
- **Date range**: 2015-01-01 to 2024-12-31
- **Parameters collected**:
  - T2M, T2M_MAX, T2M_MIN - temperature at 2m (mean, max, min)
  - RH2M - relative humidity
  - PRECTOTCORR - corrected precipitation
  - WS2M, WD2M - wind speed and direction
  - ALLSKY_SFC_SW_DWN, ALLSKY_SFC_LW_DWN, ALLSKY_SFC_PAR_TOT - all-sky radiation
  - CLRSKY_SFC_SW_DWN - clear-sky shortwave radiation
  - PS - surface pressure
  - QV2M - specific humidity
  - T2MDEW, T2MWET - dew point and wet bulb temperature

## Workflow

### 1. Data Collection and Cleaning
- Pulled daily data for each governorate and combined it into a single dataset.
- Checked for missing dates, duplicate rows, and correct daily frequency per governorate.
- Ran physical validity checks (e.g. temperature range, humidity between 0-100%, non-negative pressure and wind speed).
- Applied a seasonal-aware outlier detection method (rolling median and robust z-score) to flag anomalous temperature readings.

### 2. Exploratory Data Analysis
- Reviewed feature distributions, correlations, and missing values.
- Removed highly correlated / redundant features (T2M_MAX, T2M_MIN, T2MWET, ALLSKY_SFC_PAR_TOT, WS2M, WD2M) before modeling.
- Examined monthly and yearly temperature patterns across Egypt.

### 3. Time Series Analysis
- Built a single Egypt-wide daily T2M series by averaging across all governorates.
- Tested stationarity using ADF and KPSS tests, on the original series and on first/seasonal differenced versions.
- Used ACF/PACF plots to confirm strong annual seasonality (a spike at lag 365), which pointed toward seasonal models.

### 4. Models Tested

| Model | Description | Test MAE (°C) | Test RMSE (°C) |
|---|---|---|---|
| ARIMA(1,0,1) | Baseline, no seasonality | ~5.47 | - |
| SARIMAX + Fourier terms | ARIMA with yearly seasonality via Fourier features | ~6.86 | - |
| Seasonal Naive | Repeats the same day from the previous year | ~1.93 | - |
| LSTM (recursive forecast) | Deep learning, 365-day lookback, fully recursive multi-step forecast | ~4.11 | - |
| **LSTM (walk-forward evaluation)** | Deep learning, 365-day lookback, one-step-ahead evaluation | **~0.66** | **~0.90** |

The LSTM evaluated with walk-forward validation was the best-performing model and was selected for deployment.

### 5. Final Model
- **Architecture**: LSTM(64) -> Dropout(0.2) -> Dense(32, relu) -> Dense(1)
- **Input**: 365-day lookback window of scaled daily temperature values
- **Scaling**: MinMaxScaler fit on training data only
- **Training**: Adam optimizer, MSE loss, early stopping on validation loss

## Repository Structure

```
.
├── Task5_with_model_save.ipynb     # Full notebook: data collection, EDA, modeling, evaluation
├── egypt_weather_2015_2024_raw.csv # Raw combined dataset (all governorates)
└── models/
    ├── egypt_t2m_lstm.keras        # Trained LSTM model
    ├── egypt_t2m_scaler.pkl        # Fitted MinMaxScaler (joblib)
    └── model_config.json           # Model config (lookback window, target column, frequency)
```

## Requirements

```
pandas
numpy
requests
matplotlib
seaborn
scikit-learn
statsmodels
tensorflow
joblib
```

## Reproducing the Notebook

1. Install the requirements above.
2. Run the notebook top to bottom. It will:
   - Download raw data from the NASA POWER API for all governorates (requires internet access).
   - Save the raw dataset to `egypt_weather_2015_2024_raw.csv`.
   - Run cleaning, EDA, and stationarity/seasonality analysis.
   - Fit and evaluate ARIMA, SARIMAX+Fourier, seasonal naive, and LSTM models.
   - Save the final LSTM model, scaler, and config to the `models/` folder.

## Deployment (Django)

The saved model (`egypt_t2m_lstm.keras`), scaler (`egypt_t2m_scaler.pkl`), and config (`model_config.json`) are loaded inside a Django app to serve temperature forecasts.

General approach used for serving predictions:

1. Load the `.keras` model and the `.pkl` scaler once when the Django app starts (e.g. in `apps.py` or a dedicated model-loading module), so they are not reloaded on every request.
2. On each request, build the last 365 days of temperature history, scale it with the loaded scaler, and reshape it to `(1, 365, 1)` for the LSTM.
3. Run `model.predict(...)` and inverse-transform the output with the scaler to get the forecast in degrees Celsius.
4. Return the prediction through a Django view/endpoint (e.g. as JSON via a REST endpoint, or rendered in a template).

Adjust the exact file paths and view/URL setup above to match your Django project structure.

## Notes

- All model comparisons above were evaluated on the last 365 days (2024) of the Egypt-wide daily temperature series as the test set.
- The scaler must be fit only on training data and reused (not refit) at inference time, to avoid data leakage and keep predictions consistent with training.
