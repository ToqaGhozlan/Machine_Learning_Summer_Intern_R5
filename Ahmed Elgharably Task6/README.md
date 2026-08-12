# 🌡️ Damietta Temperature Forecast — Task 6 (Django Deployment)

A Django web app that wraps the SARIMAX(2,1,2) + annual Fourier forecasting
model trained in **Task 5** behind a small, styled form — built for Cellula
Technologies' ML Internship Program, Task 6.

> **Note on adapting the brief:** Task 6's template form assumes a model that
> takes weather features (temperature, humidity, wind speed, pressure) as
> input. That's not what Task 4/5 built. Task 4's own EDA found humidity and
> wind only weakly correlated with temperature here (r ≈ −0.1, r ≈ −0.13) —
> coastal moisture stays high year-round regardless of temperature — so a form
> asking for them wouldn't actually feed the model anything meaningful. Task 5
> is a **time-series forecaster**: it predicts a future day's temperature from
> the calendar date (via Fourier seasonal terms) and the model's own
> autoregressive structure. So the form here asks for what the model actually
> uses — a forecast date — plus a **model picker**, since Task 5's evaluation
> found the naive persistence baseline actually beats SARIMAX at long, static
> horizons on this smooth series. Showing only the "smart" model and hiding
> that would be misleading, so all three are exposed and compared honestly.

---

## What it does

- Pick a **forecast date** and a **model** (SARIMAX+Fourier, persistence, or
  seasonal-naive), optionally override the "known" starting temperature, and
  get a prediction with a 95% confidence interval (SARIMAX only) and an
  honest note about expected accuracy at that horizon.
- A sparkline of the last 30 observed days for context.
- A JSON API (`/api/predict/`) for other apps to call.
- Predictions are cached — repeat the same inputs and it returns instantly.

## Project layout

```
damietta_forecast_django/
├── manage.py
├── requirements.txt
├── config/                        # Django project (settings, urls, wsgi)
└── forecast/                      # the one app
    ├── ml.py                      # ALL model/forecast logic lives here
    ├── forms.py                   # input validation
    ├── views.py                   # form page + JSON API
    ├── urls.py
    ├── tests.py                   # 17 tests: happy paths, validation, API, caching
    ├── ml_artifacts/
    │   ├── sarimax_model.joblib   # SARIMAX(2,1,2)+Fourier(K=3), refit on full 731-day record
    │   ├── historical_temperature.csv
    │   └── model_meta.json        # order, evaluation metrics, plausibility bounds
    ├── templates/forecast/index.html
    └── static/forecast/{css,js}/
```

`ml.py` is the only file that imports statsmodels/pandas/joblib — forms and
views only ever see plain Python types. The model is loaded once, at server
startup (`forecast/apps.py`), and reused for every request via a
process-wide singleton.

## How the deployed model relates to Task 5

The notebook's own SARIMAX(2,1,2) + Fourier(K=3) specification was **refit
on the full two-year dataset** (731 days, 2024-01-01 → 2025-12-31) rather
than just the training split — standard practice once a model is selected
and evaluated, so the deployed version uses all the data it can. The
evaluation numbers shown in the app (walk-forward MAE, static-horizon MAE for
all three models) were reproduced independently from the same clean CSV and
match the notebook's stated figures:

| Model | Test setup | MAE |
|---|---|---|
| SARIMAX(2,1,2)+Fourier | 14-day walk-forward (5 folds) | ≈0.89 °C |
| SARIMAX(2,1,2)+Fourier | static 90-day held-out | ≈1.24 °C |
| Persistence | static 90-day held-out | ≈0.62 °C |
| Seasonal-naive | static 90-day held-out | ≈1.44 °C |

## Running it locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python manage.py migrate
python manage.py test forecast  # optional: run the test suite (17 tests)
python manage.py runserver
```

Open **http://127.0.0.1:8000/**.

### Try the JSON API

```bash
curl "http://127.0.0.1:8000/api/predict/?model=sarimax&date=2026-03-15"
curl "http://127.0.0.1:8000/api/predict/?model=persistence&date=2026-02-01"
curl "http://127.0.0.1:8000/api/predict/?model=seasonal_naive&date=2026-07-04&override_temp=20"
```

Valid `model` values: `sarimax`, `persistence`, `seasonal_naive`.
`date` must be after **2025-12-31** (the last day in the training data) and
no more than 365 days beyond it. Invalid input returns a `400` with a plain
JSON `{"error": "..."}` message.

## Testing it yourself

- `python manage.py test forecast` runs 17 tests covering: a valid SARIMAX
  forecast with a sane confidence interval, persistence with/without an
  override temperature, seasonal-naive matching the historical record
  exactly, every validation edge case (past date, date too far ahead, missing
  date, non-numeric temperature, out-of-range temperature, seasonal-naive
  dates with no matching prior year), and the JSON API including its caching
  behaviour.
- To sanity-check against the notebook directly: open
  `Task5_TimeSeries_Forecasting.ipynb`, and compare `sarimax_model.joblib`'s
  `.get_forecast()` output for a given date to the notebook's own SARIMAX
  fit — `forecast/ml_artifacts/model_meta.json` records the exact `order`,
  `fourier_K`, and `fourier_period` used, so the specification is identical.
- **Screenshots**: add your own `desktop.png` / `mobile.png` here after
  running the app locally and testing the form + result state, per the Task
  6 deliverables checklist.

## Deploying publicly (optional, Task 6 §7)

Before deploying anywhere (Railway, Render, etc.):

```bash
export DJANGO_DEBUG=False
export DJANGO_SECRET_KEY="<generate a real secret key>"
export DJANGO_ALLOWED_HOSTS="your-app.onrender.com"
python manage.py collectstatic
```

`sarimax_model.joblib` is ~3.4 MB — small enough to commit directly to the
repo, no external storage needed.

## Bonus challenges implemented

- ✅ **Chart of recent data** — 30-day sparkline (Chart.js) on the form page.
- ✅ **Multiple models via dropdown** — SARIMAX / persistence / seasonal-naive.
- ✅ **JSON response option** — `/api/predict/`, alongside the normal page.
- ✅ **Caching** — identical `(model, date, override_temp)` requests are
  served from Django's cache instantly (`from_cache: true` in the JSON
  response, a "served from cache" tag on the page).

---

Data: [NASA POWER API](https://power.larc.nasa.gov/). Built on Task 4 (EDA)
and Task 5 (SARIMAX/RNN forecasting) — Cellula Technologies ML Internship
Program.
