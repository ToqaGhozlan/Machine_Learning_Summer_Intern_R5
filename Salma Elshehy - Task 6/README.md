# 🌤️ Weather Prediction & Forecasting

A complete **time-series weather forecasting project** built as part of
**Task 5 -- Time Series Preprocessing & Forecasting**.

The project covers time-series preprocessing, exploratory data analysis,
stationarity testing, ARIMA/SARIMA forecasting, XGBoost forecasting,
model evaluation, and deployment with Streamlit.

## Live Demo

[![Open Live
App](https://img.shields.io/badge/Open%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://weather-prediction-model-f53pmfmwgrsuyr8eamdjzp.streamlit.app/)

## Project Overview

Weather forecasting is a time-series problem because temperature changes
over time and contains trend, seasonality, autocorrelation, and
short-term fluctuations.

The goal of this project is to build a forecasting pipeline that
prepares the weather data correctly, analyzes its temporal structure,
trains forecasting models, evaluates them using time-aware validation,
and exposes the final model through an interactive Streamlit
application.

## Objectives

-   Prepare raw weather data for time-series forecasting.
-   Resample the data to a fixed frequency.
-   Detect and handle missing values.
-   Investigate outliers.
-   Perform time-series EDA.
-   Test stationarity using ADF and KPSS.
-   Apply differencing when required.
-   Analyze ACF/PACF.
-   Build and evaluate ARIMA/SARIMA.
-   Generate forecasts with 95% confidence intervals.
-   Evaluate models using MAE, RMSE, and MAPE.
-   Use time-aware validation.
-   Deploy the forecasting application with Streamlit.
  
## Dataset

The cleaned dataset contains a date/time index and weather variables
such as:

-   Temperature
-   Humidity
-   Wind speed
-   Precipitation
-   Other available weather measurements

The data is sorted chronologically and aligned to a fixed frequency
before modeling.

## Time-Series Preprocessing

### Datetime Index & Frequency

``` python
df = pd.read_csv(
    "weather_cleaned.csv",
    parse_dates=["date"],
    index_col="date"
)

df = df.sort_index()
df = df.asfreq("D")
```

### Missing Values

Time-aware interpolation can be used for short gaps:

``` python
df["temperature"] = (
    df["temperature"]
    .interpolate(method="time", limit_direction="both")
)
```

Long gaps should be handled using an appropriate seasonal strategy
rather than blindly forward-filling temperature.

### Outlier Detection

Weather observations are inspected using:

-   Rolling z-score
-   Rolling IQR
-   Domain-based physical limits
-   Time-series plots
-   Boxplots

The objective is to preserve valid weather events and maintain a
continuous series.

## Exploratory Data Analysis

The EDA stage examines:

-   Temperature trend
-   Daily/yearly seasonality
-   Missing timestamps
-   Missing values
-   Temperature distribution
-   Outliers
-   Correlations between temperature, humidity, wind, precipitation, and
    pressure when available

## Stationarity

ARIMA/SARIMA requires the target series to become stationary after
differencing.

### ADF

-   `p-value < 0.05` → evidence supporting stationarity.
-   `p-value >= 0.05` → insufficient evidence of stationarity.

### KPSS

-   `p-value < 0.05` → evidence against stationarity.
-   `p-value >= 0.05` → consistent with stationarity.

Both tests are useful because they approach stationarity from different
null hypotheses.

## Differencing

If required, first-order and seasonal differencing can be investigated:

``` python
df["temp_diff1"] = df["temperature"].diff(1)
df["temp_diff_seasonal"] = df["temperature"].diff(24)
```

For hourly data, `24` represents daily seasonality. The appropriate
seasonal period should match the actual data frequency.

## ACF & PACF

ACF and PACF plots help propose:

``` text
(p, d, q)
(P, D, Q, m)
```

The manually selected orders should be compared with automated model
selection.

## SARIMA

The main statistical forecasting approach is SARIMA:

``` text
SARIMA(p,d,q)(P,D,Q,m)
```

Example:

``` python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    train["temperature"],
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
)

fit = model.fit(disp=False)
```

The final order should be justified using stationarity tests, ACF/PACF,
automated search, and residual diagnostics.

## Auto ARIMA

``` python
import pmdarima as pm

auto_model = pm.auto_arima(
    train["temperature"],
    seasonal=True,
    m=24,
    stepwise=True,
    trace=True,
    suppress_warnings=True,
    error_action="ignore"
)
```

The auto-selected model is compared with the manually proposed SARIMA
configuration.

## Residual Diagnostics

Residuals should ideally:

-   Have approximately zero mean.
-   Show no obvious trend.
-   Have no significant remaining autocorrelation.
-   Behave approximately like white noise.

Useful diagnostics include:

``` python
fit.plot_diagnostics(figsize=(10, 8))
```

and the Ljung-Box test.

## Confidence Intervals

SARIMA produces both forecasts and uncertainty intervals:

``` python
forecast = fit.get_forecast(steps=len(test))

pred_mean = forecast.predicted_mean

confidence_interval = forecast.conf_int(alpha=0.05)
```

The application can display the forecast together with the 95%
confidence interval.

## XGBoost Forecasting

XGBoost is included as a machine-learning comparison.

Because XGBoost does not inherently understand temporal order, time
information is represented through engineered features.

### Calendar Features

``` text
year
month
day
day_of_week
```

### Lag Features

``` text
temperature lag 1
temperature lag 7
humidity lag 1
wind speed lag 1
```

### Rolling Features

``` text
3-day temperature rolling mean
7-day temperature rolling mean
```

### External Features

``` text
humidity
wind_speed
precipitation
```

These features allow XGBoost to learn nonlinear relationships between
recent weather conditions and temperature.

## Recursive Future Forecasting

For future dates, actual temperature is unknown. Therefore, multi-step
XGBoost forecasting should be performed recursively:

1.  Predict the next temperature.
2.  Add that prediction to the historical sequence.
3.  Recalculate lag/rolling features.
4.  Predict the following day.
5.  Repeat for the requested forecast horizon.

Future external variables such as humidity and wind speed should ideally
come from a real weather forecast/API. If they are simulated, this
assumption should be clearly documented.

## Evaluation

Models are evaluated on a chronological held-out test set.

### MAE

Mean Absolute Error. Lower is better.

### RMSE

Root Mean Squared Error. Penalizes larger errors more strongly.

### MAPE

Mean Absolute Percentage Error. It should be interpreted carefully when
actual temperatures can approach zero.

Example:

``` python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

mape = np.mean(
    np.abs((y_test - predictions) / y_test)
) * 100
```

## Time-Aware Validation

Random shuffling must be avoided for time-series evaluation.

A rolling-origin strategy can be used:

``` python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(
    n_splits=5,
    test_size=24
)
```

This better represents how the model will behave when forecasting
genuinely unseen future observations.

## Streamlit Application

The deployed application provides an interactive interface where users
can:

1.  Select the forecast horizon.
2.  Generate future temperature predictions.
3.  View the predicted temperatures.
4.  Visualize the future temperature trend.

## Technologies

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Plotly
-   Scikit-learn
-   Statsmodels
-   pmdarima
-   Joblib
-   Streamlit

## Key Takeaways

-   SARIMA is designed specifically to model autocorrelation and
    seasonality in time series.
-   XGBoost can capture nonlinear relationships when temporal
    information is represented using lag, rolling, calendar, and
    external features.
-   Future external variables must be available or estimated without
    using actual future observations.
-   A visually smooth forecast is not automatically an accurate
    forecast.
-   Final model selection should be based on time-aware evaluation
    metrics.

## Author

**Salma Elshehy**\
Computer & Systems Engineering\
Alexandria University
