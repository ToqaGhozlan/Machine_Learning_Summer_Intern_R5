# Uber Fare Prediction — Flask App

## What this is
A Flask web app that takes trip details (pickup/dropoff location, date/time,
passenger count, car condition, weather, traffic) and returns a predicted fare
using the Random Forest pipeline trained in Task 2.

## Model Comparison & Selection

| Model                        | MAE      | RMSE     | R²       |
|-------------------------------|----------|----------|----------|
| Baseline (Linear Regression)  | 2.151853 | 4.037306 | 0.820000 |
| Tuned (Random Forest)         | 1.677047 | 3.347682 | 0.876241 |

The tuned Random Forest was chosen for deployment. It beats the Linear
Regression baseline on every test-set metric: MAE drops from 2.15 to 1.68
(~22% lower average error), RMSE drops from 4.04 to 3.35 (fewer large
misses), and R² rises from 0.82 to 0.88 (explains more of the variance in
fare amount). Since the improvement holds consistently across all three
metrics rather than trading one off against another, the decision isn't
close enough to need a speed/interpretability tie-breaker — Random Forest
is simply the better model on the evidence.

The fitted pipeline (preprocessing + tuned Random Forest bundled together)
was then saved with `joblib`:

```python
import joblib

joblib.dump(best_rf_pipeline, 'uber_fare_pipeline.pkl')
print("Pipeline saved as uber_fare_pipeline.pkl")
```

This is why `app.py` can call `model.predict()` directly on raw feature
values without a separate scaling/encoding step — those steps live inside
the saved pipeline itself.

## Folder structure
```
Code/
├── app.py
├── uber_fare_pipeline_small.pkl
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── tailwind-config.js
│       └── app.js
└── README.md
```

## How to run locally

1. Make sure your virtual environment is active (the one with scikit-learn 1.6.1):
   ```
   .\venv\Scripts\Activate
   ```

2. Make sure Flask is installed in this environment:
   ```
   pip install flask joblib pandas numpy scikit-learn
   ```

3. Confirm `uber_fare_pipeline_small.pkl` is in the same folder as `app.py`.

4. Run the app:
   ```
   python app.py
   ```

5. Open your browser and go to:
   ```
   http://127.0.0.1:5000
   ```

6. Fill in the form and click "Predict Fare".

## Notes
- All fields are required. Latitude/longitude are restricted to NYC's
  approximate bounds; passenger count is restricted to 1-6, matching the
  ranges used during Task 2's data cleaning.
- If any field is missing, invalid, or out of range, the app shows an error
  message instead of crashing or producing a silent wrong prediction.
- The app recreates every engineered feature from Task 2 (distance to
  dropoff, distance to JFK/EWR/LGA airports, bearing, hour/day/month/weekday/
  year, and the airport-trip flag) from the raw form inputs before calling
  the model — the same pipeline logic used during training.
- The UI is split into `templates/index.html` (markup + Jinja logic),
  `static/css/style.css` (custom styling), `static/js/tailwind-config.js`
  (Tailwind theme config), and `static/js/app.js` (map interaction, form
  behavior, and live distance calculation). Flask serves `static/` and
  `templates/` automatically, so no changes to `app.py` were needed for
  this split.
