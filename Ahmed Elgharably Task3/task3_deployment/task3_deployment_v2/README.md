# Task 3 — Uber Fare Prediction: Model Deployment (v2)

## What changed from the first version

1. **Fixed a real correctness risk in the feature formulas.** `distance`, `bearing`, and
   `nyc_dist` are *pre-engineered* columns that already existed in the dataset before Task 2 —
   nobody computed them from raw coordinates in this project, so the exact formula/unit wasn't
   directly documented. V1 guessed miles. Two independent signals now point to **kilometres**
   (Earth radius 6371) with a specific bearing sign convention instead — see
   `feature_engineering.py`'s docstring. **`validate_formulas.py`** lets you *prove* this on
   your own copy of `final_internship_data.csv` in one command, instead of trusting a guess.
2. **Single source of truth**: feature engineering now lives in its own `feature_engineering.py`
   module, imported by both `app.py` and (optionally) your training notebook — instead of the
   logic being duplicated inline in `app.py` only.
3. **Interactive map** (Leaflet + OpenStreetMap) to pick pickup/dropoff points by clicking,
   instead of typing raw latitude/longitude by hand.
4. **AJAX submission** — the predicted fare appears on the same page without a full reload.
5. **JSON API support** — `/predict` accepts both an HTML form post and a JSON body, so it can
   be tested with `curl`/Postman or wired into another frontend later.
6. **Automated tests** (`tests/test_app.py`, pytest) covering the happy path and every invalid-
   input case, so you can verify the app logic in seconds without opening a browser.
7. `/health` endpoint, structured logging, `requirements.txt`.

## What's in this folder

| File | Purpose |
|---|---|
| `Task3_Model_Deployment.ipynb` | Task 2 notebook + Part A: trains a 3rd model (Random Forest), builds the final 3-model comparison table, writes the model justification, saves `fare_amount_pipeline.joblib`. |
| `feature_engineering.py` | **Single source of truth.** Raw trip input → the exact feature row the model was trained on. Validated haversine/bearing formulas live here. |
| `validate_formulas.py` | Run once against your real CSV to confirm/adjust the formula assumptions above. |
| `app.py` | Flask backend — loads the pipeline once, validates input, calls `feature_engineering`, returns a prediction (HTML or JSON). |
| `templates/index.html` | Map + form + result/error UI (single page). |
| `static/js/app.js` | Map click handling + AJAX submit. |
| `static/css/style.css` | Styling. |
| `tests/test_app.py` | Automated smoke tests (routing/validation/error-handling) using a dummy schema-matching pipeline — doesn't need the real dataset. |
| `requirements.txt` | Dependencies. |
| `fare_amount_pipeline.joblib` | **You generate this** by running the notebook (below) — the fitted `Pipeline` `app.py` loads. |

## 1. Confirm the formula assumption (recommended, one-time)

```bash
pip install -r requirements.txt
python validate_formulas.py "C:\path\to\final_internship_data.csv"
```

This prints an MAE for km vs. miles and for both bearing sign conventions against your real
data. Whichever line comes out near-zero is correct. If it's not km / the current bearing
sign, open `feature_engineering.py` and update `EARTH_RADIUS_KM` and the `dlon` line in
`bearing_rad()` to match — everything else (`app.py`, tests) stays the same.

## 2. Run the notebook to produce the model file

```bash
jupyter notebook Task3_Model_Deployment.ipynb
```

Fix the `pd.read_csv(...)` path in the first cell, then Run All. The new **Part A** section
trains the Random Forest, prints the final 3-model comparison table, and re-confirms the
justification for deploying the tuned `HistGradientBoostingRegressor`. The existing save cell
then writes `fare_amount_pipeline.joblib` — copy it into this same folder, next to `app.py`.

## 3. Run the automated tests (optional but recommended)

```bash
pytest tests/test_app.py -v
```

These don't need the real model — they use a tiny dummy pipeline with the same schema, purely
to prove the Flask app's routing/validation/error handling is correct before you screenshot it.

## 4. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** — click the map once for pickup (green marker), again for
dropoff (red marker), fill in the rest, and submit.

## 5. Testing for submission (Part D)

Through the actual browser UI:
- **3 valid trips** with different pickup/dropoff points, times, and conditions → confirm a
  sensible `$` prediction appears in the green banner.
- **1 invalid case** — e.g. leave a field empty, or click a point far outside NYC — confirm the
  red banner lists a clear message instead of the app crashing or silently mispredicting.

Screenshot the filled-in form *and* the resulting page for each case.

## Notes

- The model is loaded **once** at import time (`model = joblib.load(...)` at module load), not
  inside `/predict`, so repeated requests don't reload or retrain anything.
- All validation happens server-side in `feature_engineering.validate_raw_input`, so a
  malformed or scripted request is still caught even if someone bypasses the browser's own
  `required`/`min`/`max` attributes.
- `distance` is capped at 100 (same fixed winsorizing threshold Task 2 used) inside
  `build_feature_row`, so an unusually long trip won't silently feed the model an out-of-
  distribution value.
