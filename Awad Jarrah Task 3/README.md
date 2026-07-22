# Uber Fare Predictor — Flask App

A small Flask web app that serves fare predictions from a trained Random
Forest pipeline (tuned via `RandomizedSearchCV`, MAE ≈ $1.52 on the held-out
test set — see the notebook for the full model comparison).

## Project structure

```
app.py
feature_engineering.py
uber_fare_pipeline.joblib
requirements.txt
templates/
    index.html
static/
    css/
        variables.css
        base.css
        layout.css
        components.css
    js/
        main.js
        distance-preview.js
        form-validation.js
        submit-state.js
        fare-reveal.js
```

## Setup

1. **Use Python 3.10+** and create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   `requirements.txt` pins `scikit-learn==1.9.0`. This **must match the
   scikit-learn version the notebook used to train and save
   `uber_fare_pipeline.joblib`** — a version mismatch can raise warnings or
   errors when the pipeline is unpickled. Check the training environment
   with:
   ```python
   import sklearn; print(sklearn.__version__)
   ```
   If it doesn't say `1.9.0`, either re-run the notebook's training/saving
   cells in a `1.9.0` environment, or change the pin in `requirements.txt`
   to match whatever version was actually used to train.

3. **Make sure `uber_fare_pipeline.joblib` is present** in the same folder
   as `app.py` (produced by the training notebook's `joblib.dump(...)`
   step). The app will refuse to start with a clear error if it's missing.

## Run

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

## Using the app

Fill in the trip's pickup date/time, pickup and dropoff coordinates
(decimal degrees), passenger count, and the car condition / weather /
traffic dropdowns, then click **Predict fare**. A live straight-line
distance preview appears as you fill in the coordinates; the predicted
fare appears at the bottom of the form after submitting.

Invalid or missing input (empty required fields, out-of-range coordinates,
non-numeric values, etc.) shows a clear error message on the same page
instead of crashing — see `feature_engineering.py`'s `validate_raw_input()`
for the exact rules.

## Sanity-checking the deployment

`sanity_check.py` (if included) re-runs the exact same
`preprocess_and_predict()` pipeline the app uses against real rows from
`final_internship_data.csv`, plus a few hand-checkable edge cases (a
zero-distance trip, a known Manhattan→JFK trip, and an invalid-input
case). Run it with:
```bash
python sanity_check.py
```
