# Algiers Weather Forecaster — Django Deployment (Task 6)

**ML Internship Program @ Cellula Technologies** — Task 6: wraps the Task 5 SARIMAX
forecasting model in a Django web app with a clean Bootstrap 5 UI, form validation,
a JSON API, and response caching.

## A necessary adaptation — please read first

The assignment's example inputs ("temperature, humidity, wind speed, pressure") describe a
**tabular regression** model. The model actually produced in Task 5 is a **time-series
forecaster** (`SARIMAX(2,1,2)` + Fourier terms for the annual seasonal cycle) — it doesn't
take current weather readings as input at all; it takes a **future date** and extrapolates
from the learned trend + seasonal pattern. Building a form that asked for "today's humidity"
would be asking for information the model was never trained to use and can't actually
consume. So the form here collects what this model *does* need — a target date — validated
against the model's actual supported forecast range. This is explained in more depth in
`predictor/ml_model.py`'s docstring.

## What's included

```
weather_deploy/          Django project (settings, urls)
predictor/                App: ml_model.py, forms.py, views.py, urls.py, templates/
static/                   Custom CSS + self-hosted Bootstrap/FontAwesome/Chart.js
weather_model.pkl         Serialized SARIMAX model (joblib), refit on the full 5-year series
prepare_model.py          Script that produced weather_model.pkl (rerun to regenerate it)
run_tests.py              Functional test suite (10 tests) exercising the whole app
screenshots/              Desktop form / result / validation-error / mobile screenshots
requirements.txt
```

## Running it locally

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Then open `http://127.0.0.1:8000/`. The model file (`weather_model.pkl`) is already included
and loaded once at process start (`predictor/ml_model.py`'s module-level singleton) — no
retraining needed to run the app. To regenerate it from scratch: `python prepare_model.py`.

## What it does

- Enter a date within the model's supported range (the page shows the exact valid window).
- Get back a predicted temperature, a color/icon-coded category (Cold/Mild/Warm/Hot), and a
  95% confidence interval.
- A chart of the last 30 days of actual training data sits alongside, for context.
- A model-performance section reports MAE, RMSE and MAPE on a 180-day chronological holdout and compares SARIMAX with a persistence baseline.
- An actual-vs-predicted chart visualizes performance on unseen test data.

## Design checklist (per the assignment's UI requirements)

- [x] Clear title + one-line description of what the app does
- [x] Card-style form with a labeled input
- [x] Inline feedback for invalid values (empty field, bad format, out-of-range date)
- [x] Visually distinct, color-coded result panel with an icon
- [x] Responsive layout — verified at both desktop (1280px) and mobile (390px) widths
- [x] Consistent palette/spacing throughout — custom CSS on top of Bootstrap 5, no
      unstyled default form elements

## Edge cases handled

| Input | Behavior |
|---|---|
| Empty field | "Please choose a date." |
| Non-date text | "That doesn't look like a valid date." |
| Date before the model's training end | "Choose a date on or after \<date\> — the model's history ends \<date\>." |
| Date more than 180 days past training end | "Choose a date on or before \<date\> — forecasts beyond 180 days out are too uncertain..." |
| All of the above via the JSON API | Returns HTTP 400 with a JSON `error` field instead of a 500 |

## Bonus challenges implemented

- **Chart of recent inputs** — Chart.js line chart of the last 30 days of training data.
- **JSON API** — `GET /api/forecast/?date=YYYY-MM-DD`, alongside the normal HTML page, for
  other apps to consume.
- **Caching** — identical date requests (via the form or the API) are served from Django's
  cache framework on repeat, flagged in the response (`from_cache: true` / a "served from
  cache" badge on the page).
- *(Not implemented: multiple-model dropdown — only one model was trained in Task 5, so
  there's nothing to switch between yet.)*

## Model evaluation

The deployed model is refit on the full 2021-01-01 to 2025-12-31 history. For an honest performance estimate, `evaluate_model.py` holds out the final 180 days, trains the same SARIMAX(2,1,2) + Fourier(order=3) model on the earlier observations, and evaluates it on the unseen holdout. It also compares against a simple persistence baseline (future temperature equals the last observed training value).

Current deterministic evaluation results:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| SARIMAX + Fourier | 1.023°C | 1.302°C | 5.49% |
| Persistence baseline | 7.550°C | 9.469°C | 50.11% |

To regenerate the evaluation after changing the data/model, run:

```bash
python evaluate_model.py
```

The results are displayed in the Django UI and are also available at `GET /api/evaluation/`.

## Testing

`run_tests.py` uses Django's test client to exercise the full app without needing a manual
browser session — run it with `python run_tests.py` after `pip install -r requirements.txt`.
It covers: page rendering, a valid prediction, all four edge cases above, both JSON API
paths, cache behavior, and a direct cross-check confirming the web app's prediction exactly
matches calling the model directly (the same check the rubric's "Model Integration" line
asks for) — all 10 checks pass.

## Screenshots

See `screenshots/` — form (empty), result (valid prediction), validation error, and mobile
view. Captured against the running app with a headless browser, not mocked.

## Deploying publicly (optional, per the assignment)

Before deploying to Railway/Render/etc.:
1. Set environment variables: `DJANGO_DEBUG=False` and `DJANGO_ALLOWED_HOSTS=your-domain.com`
   (both are already read from the environment in `settings.py` — no code change needed).
2. Set `DJANGO_SECRET_KEY` to a real secret (a fallback dev key is used otherwise — fine for
   local testing, not for production).
3. Run `python manage.py collectstatic` so `STATIC_ROOT` is populated for your production
   static-file server (all CSS/JS/fonts are already vendored locally in `static/vendor/` —
   no external CDN dependency, so nothing extra to fetch at deploy time).
4. `weather_model.pkl` is ~8.5 MB — small enough to commit directly to the repo.

## Tools

Django · statsmodels (SARIMAX) · joblib · Bootstrap 5 · Chart.js · FontAwesome
