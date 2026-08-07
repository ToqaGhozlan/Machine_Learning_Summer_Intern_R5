# Uber Fare Prediction — Task 3: Model Deployment

## Overview
This project trains, compares, and deploys a regression model that predicts Uber trip fares.
The final model (Gradient Boosting Regressor) is served through a Flask web application
with a simple HTML form for entering trip details.

## Project Structure
ML/
├── EDA.ipynb # Task 1 - Exploratory Data Analysis
├── Preprocessing.ipynb # Task 2 - Data cleaning & preprocessing
├── Task 3.ipynb # Task 3 - Model comparison & training
├── final_internship_data.csv # Raw dataset
├── uber_fare_pipeline.pkl # Saved sklearn Pipeline (preprocessor + model)
├── app.py # Flask application (backend)
├── templates/
│ ├── index.html # Input form
│ └── result.html # Prediction result page
├── static/
│ └── style.css # Styling
└── screenshots/ # Testing evidence (Part D)

## Model Comparison (Part A)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Gradient Boosting | 1.918 | 3.789 | 0.821 |
| Random Forest | 2.233 | 4.233 | 0.777 |
| Decision Tree | 2.657 | 4.749 | 0.720 |
| Linear Regression | 2.964 | 6.730 | 0.437 |

**Chosen model: Gradient Boosting Regressor** — best performance on every metric (lowest
MAE/RMSE, highest R²), and fast inference time (~1.08ms per 595 rows). See `Task 3.ipynb`
for the full justification.

## Important Notes on the Data Pipeline
- Coordinates in the original dataset (`final_internship_data.csv`) are stored in **radians**.
  The Flask app accepts coordinates in normal **degrees** from the user and converts them to
  radians internally before prediction.
- `distance`, `bearing`, and `jfk_dist` are not asked from the user directly. They are
  calculated inside `app.py` using the Haversine formula (distance) and a standard bearing
  formula, based on the pickup/dropoff coordinates the user enters. The original formulas
  used to generate these columns in the source dataset were not available, so standard
  formulas were used as the closest reasonable approximation.
- The target (`fare_amount`) was trained on a log-transformed scale (`log1p`). Predictions
  are converted back with `expm1` before being shown to the user.
- `weekday` uses Python's standard `datetime.weekday()` convention (0 = Monday, 6 = Sunday).

## How to Run the App Locally

1. Make sure the following files are in the same folder: `app.py`, `uber_fare_pipeline.pkl`,
   `templates/`, `static/`.

2. Install the required packages:
```bash
   pip install flask numpy pandas scikit-learn==1.7.2 joblib
```
   > Note: scikit-learn version must match the version used to save `uber_fare_pipeline.pkl`
   > to avoid `InconsistentVersionWarning` / unpickling errors.

3. Run the app:
```bash
   python app.py
```

4. Open your browser at: http://127.0.0.1:5000

5. Fill in the trip details and click "احسب الأجرة المتوقعة" (Calculate Predicted Fare)
   to get a prediction.

## Testing (Part D)
Three successful predictions and one invalid-input case were tested through the actual UI.
Screenshots are available in the `screenshots/` folder:
- `case1_form.png` / `case1_result.png` — short trip in Manhattan → $6.40
- `case2_form.png` / `case2_result.png` — trip to JFK Airport → $48.09
- `case3_form.png` / `case3_result.png` — medium trip, max passengers → $13.85
- `case4_invalid_form.png` / `case4_invalid_result.png` — out-of-range latitude,
  rejected with a clear error message instead of crashing or producing a silent
  wrong prediction.