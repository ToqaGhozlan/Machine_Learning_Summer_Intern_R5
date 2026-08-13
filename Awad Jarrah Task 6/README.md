# Weather Temperature Predictor (Django)

A Django web app that wraps the three forecasting models trained in Task 5
(`arma_manual`, `armax_auto`, `fourier_auto`) behind a simple, styled form —
pick a model, enter today's weather, get an instant temperature prediction.

## Project layout

```
weather_predictor/
├── manage.py
├── requirements.txt
├── saved_models/              <- put your .pkl files here (see saved_models/README.md)
│   ├── arma_manual.pkl
│   ├── armax_auto.pkl
│   └── fourier_auto.pkl
├── weather_predictor/         <- Django project config
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
└── predictor/                 <- the actual app
    ├── ml_models.py           <- loads pickles, runs predictions
    ├── forms.py                <- input validation
    ├── views.py                <- HTML view + /api/predict/ JSON view
    ├── urls.py
    ├── templates/predictor/    <- Bootstrap 5 UI
    └── static/predictor/css/style.css
```

## 1. Setup

```bash
cd weather_predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Add your trained models

Copy the three pickle files you saved in Task 5 into `saved_models/`:

```
saved_models/arma_manual.pkl
saved_models/armax_auto.pkl
saved_models/fourier_auto.pkl
```

> If a file is missing, the app still runs — it just shows a warning banner
> and disables that specific model instead of crashing.

## 3. Run it locally

```bash
python manage.py migrate
python manage.py createsuperuser   # optional, to log into /admin/
python manage.py runserver
```

Open **http://127.0.0.1:8000/** in your browser. Every prediction you make is
logged (model used, predicted temp, confidence range, timestamp), shown as a
chart under the form, and browsable at **http://127.0.0.1:8000/admin/**
(under Predictor → Predictions) if you created a superuser.

## How each model is used

| Model | Inputs needed | What happens |
|---|---|---|
| Manual ARMA | a forecast date | The date is converted internally to "days past 2025-07-31" and passed to `get_forecast(steps=...)` — no weather inputs, since this model has no exogenous covariates. |
| Auto ARMAX | humidity, tempmax, dew, tempmin, precip, cloudcover, precipprob | Predicts the next day's temperature from those same-day covariates. No date field — the model was never trained with any date-derived features, so a date wouldn't change anything. |
| Auto ARMAX + Fourier (best model) | the 7 covariates above + a forecast date | Same as ARMAX, plus seasonal (annual) terms computed from the date, matching how it was trained. |

The form only shows the fields relevant to whichever model you pick.

## JSON API (bonus)

`GET` or `POST` the same field names to `/api/predict/`:

```bash
curl "http://127.0.0.1:8000/api/predict/?model_choice=arma_manual&forecast_date=2025-08-05"
```

```json
{"ok": true, "cached": false, "result": {"predicted_temp": 24.3, "lower": 21.1, "upper": 27.5, "detail": "..."}}
```

Validation errors come back as HTTP 400 with an `errors` object.

## Caching

Identical form submissions (same model + same inputs) are cached in memory
for 1 hour, so a repeated request returns instantly instead of re-running
the model — this shows up as a "served from cache" badge on the result card.

## Deploying publicly (optional)

1. Set environment variables instead of the dev defaults in `settings.py`:
   - `DJANGO_DEBUG=False`
   - `DJANGO_SECRET_KEY=<something random>`
   - `DJANGO_ALLOWED_HOSTS=yourdomain.com`
2. Run `python manage.py collectstatic` and make sure your host serves
   `staticfiles/` (or use something like WhiteNoise).
3. Make sure the three `.pkl` files are included in the deployment (they're
   small enough to commit; if not, load them from external storage at
   startup instead).
4. Platforms like Railway or Render can run this with minimal extra config
   since it's a standard Django + SQLite app.

## Running tests

```bash
python manage.py test
```
