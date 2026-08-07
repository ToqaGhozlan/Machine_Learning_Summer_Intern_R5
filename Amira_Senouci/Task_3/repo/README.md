# Uber Fare Prediction — NYC

**ML Internship Program @ Cellula Technologies** — end-to-end regression project on 500,000 NYC Uber ride records (2009–2015): from raw data to a deployed-ready scikit-learn pipeline that prices trips within **$1.67 on average (R² 0.86)** on a fully held-out year.

## Repository structure

```
task1/  Exploratory Data Analysis        → Uber_Fare_EDA.ipynb (18 figures)
task2/  Preprocessing & Modeling         → Uber_Fare_Preprocessing.ipynb
        Fitted pipeline (joblib)         → uber_fare_pipeline.joblib
```

## Task 1 — Exploratory Data Analysis

Every plot choice justified, every cleaning decision documented. Highlights:

- **Distance dominates fare** (r = 0.86), and the pricing formula is affine: $/km falls from $13.10 on sub-500 m hops to ~$2.50 on long trips — the signature of a base fare amortized over distance.
- **Three categorical columns are synthetic noise** (`weather`, `traffic_condition`, `car_condition`): perfectly uniform level counts, identical fare distributions.
- **Real pricing events are visible in the data**: the September 2012 NYC fare increase appears as a step in the yearly trend, and the $52 JFK flat fare shows up as a horizontal shelf among long trips — starting exactly in 2012.
- Coordinates arrive in **radians** despite documentation claiming degrees — verified and converted; 3.5% of rows removed with physical justification for each rule.

![Fare vs distance](task1/assets/fare_vs_distance.png)

## Task 2 — Preprocessing & Modeling

Evidence-driven preprocessing (MCAR analysis, encoding decided by **controlled ablation**, empirical scaler comparison, outliers kept with a log-target absorbing their leverage), a **chronological split** (train 2009–2014, test 2015 — no future leakage), and two models wrapped in reusable sklearn Pipelines:

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Linear Regression (baseline) | 2.36 | 5.33 | 0.785 |
| HistGradientBoosting (default) | 1.85 | 4.67 | 0.835 |
| **HistGradientBoosting (tuned)** | **1.67** | **4.32** | **0.859** |

Tuned via `RandomizedSearchCV` (20 candidates × 3 folds). Permutation importance ranks `drop_jfk_dist` — a feature **engineered from Task 1's flat-fare finding** (haversine distance from drop-off to JFK) — as the **#2 predictor**, validating the EDA → feature-engineering loop.

### Using the trained pipeline

```python
import joblib
pipeline = joblib.load("task2/uber_fare_pipeline.joblib")
predictions = pipeline.predict(feature_frame)  # dollars, no manual preprocessing
```

## Reproducing

The dataset (`final_internship_data.csv`, 170 MB) exceeds GitHub's file-size limit and is not included — download it from the assignment's Google Drive link, place a copy next to each notebook, then:

```bash
pip install -r requirements.txt
jupyter notebook
```

## Tools

Python · pandas · NumPy · Matplotlib · Seaborn · SciPy · scikit-learn · joblib
