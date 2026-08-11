# Cairo Weather Forecast — Model Deployment

## Part A — Model Comparison

### Temperature Forecasting

Two models were built and evaluated for forecasting Cairo's daily mean temperature (`temp_mean`):

| Model | MAE (°C) | RMSE (°C) | MAPE (%) | Notes |
|---|---|---|---|---|
| SARIMA + Fourier terms (K=1) | 1.27 | 1.60 | 7.89 | m=7 seasonal order + yearly Fourier terms as exogenous regressors, to work around the computational cost of native m=365 seasonality |
| **ETS (Holt-Winters, m=365)** | **1.21** | **1.53** | **6.70** | Native yearly seasonality, no workaround needed |

**Chosen model: ETS (Holt-Winters, m=365).**

ETS outperformed SARIMA+Fourier on MAE, RMSE, and MAPE while using the true
yearly seasonal period directly, rather than approximating it with exogenous
Fourier terms on top of a short (m=7) seasonal order. It is also
computationally lighter to fit and simpler to deploy, since it only requires
a single input — the number of days ahead to forecast — with no exogenous
features to reconstruct at prediction time. Given it wins on both accuracy
and simplicity, ETS was selected as the production model.

### Humidity Forecasting

As an extension beyond the core task, an ETS (Holt-Winters, m=365) model was
also built for daily humidity (`humidity`), using the same architecture as
the temperature model, to support a combined weather forecast in the app.

| Model | MAE (%) | RMSE (%) |
|---|---|---|
| ETS (Holt-Winters, m=365) | 7.70 | 10.20 |

Humidity is inherently noisier than temperature (more day-to-day variation),
so its error is proportionally higher. ETS was kept as the deployed model
for consistency with the temperature model — both use the same date-in,
forecast-out interface, with no additional input features required at
prediction time, which keeps the app's pipeline simple.

## Part B — Saved Models

Both final models were retrained on the full 5-year dataset (2020–2024,
train + test combined) before saving, so the deployed models benefit from
all available historical data rather than the held-out test portion used
during evaluation.

| File | Contents |
|---|---|
| `ets_temp_model.pkl` | Fitted ETS model for `temp_mean`, trained on full dataset |
| `ets_humidity_model.pkl` | Fitted ETS model for `humidity`, trained on full dataset |
| `last_known_date.json` | The last date present in the training data — used to compute how many days ahead to forecast for any user-requested date |

Both models are loaded via `joblib.load(...)`. Prediction for a given future
date is computed as:

```python
days_ahead = (requested_date - last_known_date).days
temp_pred = ets_temp_model.forecast(steps=days_ahead).iloc[-1]
humidity_pred = ets_humidity_model.forecast(steps=days_ahead).iloc[-1]
```

No additional preprocessing or feature engineering is required at prediction
time — both models take only the target date as input, keeping the
pipeline simple and self-contained.

## UI Overview

The interface is a single-page form built with **Tailwind CSS** (loaded via
CDN, no build step required) using a custom color and type-scale
configuration for a consistent look, rather than Bootstrap defaults.

- **Form card:** a centered, shadowed card with a single date input
  (calendar icon, native browser date picker)
- **Inline validation:** invalid or missing dates highlight the input
  border in red and show an inline error message with an icon, without
  losing the previously entered value
- **Result panel:** on a successful forecast, a two-column result panel
  appears below the form showing temperature and humidity, each in its
  own color-coded card with a matching icon
- **Responsive layout:** the form is capped at a max width and centered,
  with spacing/typography that scales down on mobile via Tailwind's `md:`
  breakpoint prefixes

*(See screenshots below for the form and result states.)*

## Input Validation & Error Handling

All validation happens server-side in `views.py` before any forecast is run:

| Case | Behavior |
|---|---|
| Empty date field | "Please enter a date." |
| Malformed / non-date input | "That doesn't look like a valid date. Please use the date picker." |
| Date on or before the training data's last date | Rejected with the cutoff date shown in the message |
| Date more than ~10 years ahead | Rejected — ETS forecasts that far out are no longer meaningful |

On any error, the form re-renders with the error message inline next to
the date field, and the previously submitted date is preserved in the
input rather than being cleared.

## Project Structure

```
Code/
├── README.md
└── weather_forecast/            <- Django project root
    ├── manage.py
    ├── weather_forecast/         <- project settings
    │   ├── settings.py
    │   ├── urls.py
    │   └── ...
    └── forecast/                 <- app
        ├── urls.py
        ├── views.py               <- loads models, handles predictions
        ├── models_data/           <- saved model files
        │   ├── ets_temp_model.pkl
        │   ├── ets_humidity_model.pkl
        │   └── last_known_date.json
        └── templates/
            └── forecast/
                └── form.html      <- the form + results page
```

## How to Run the App Locally

1. **Install dependencies:**
   ```bash
   pip install django joblib statsmodels pandas numpy
   ```

2. **Start the development server** (from the `weather_forecast` folder, the one containing `manage.py`):
   ```bash
   python manage.py runserver
   ```

3. **Open the app** in your browser:
   ```
   http://127.0.0.1:8000/
   ```

4. **Use the app:** enter a date and click "Get Forecast" to see the predicted
   temperature and humidity for that date in Cairo.

No database setup or migrations are required — this app has no models or
user accounts, it only loads the saved `.pkl` forecasting models on startup.
