# Uber Fare Prediction — Flask Web Application

## Overview
This Flask app deploys a trained machine-learning model (HistGradientBoostingRegressor) that predicts Uber fare amounts based on trip details. The model was trained in Task 2 using a full sklearn pipeline that includes data imputation, scaling, ordinal/one-hot encoding, Lasso feature selection, and gradient boosting regression.

## Prerequisites
- Python 3.10+
- The trained pipeline file: `uber_fare_pipeline.joblib` (produced in Task 2)

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Flask server:**
   ```bash
   python app.py
   ```

3. **Open the app in your browser:**
   ```
   http://localhost:5000
   ```

## Using the App

Fill in the form with:
- **Pickup / Dropoff coordinates** — latitude and longitude in degrees (e.g., 40.7580, −73.9855 for Times Square)
- **Pickup Date & Time** — when the trip starts
- **Passenger Count** — 1 to 6
- **Car Condition** — Bad / Good / Very Good / Excellent
- **Weather** — Windy / Cloudy / Stormy / Sunny / Rainy
- **Traffic Condition** — Flow Traffic / Dense Traffic / Congested Traffic

Click **Predict Fare** to see the estimated fare amount.

## Project Structure
```
task3/
├── app.py                    # Flask backend
├── templates/
│   └── index.html            # Web UI template
├── static/
│   └── style.css             # Styling
├── uber_fare_pipeline.joblib # Trained model pipeline
├── Task2_Preprocessing.ipynb # Task 2 notebook (preprocessing + training)
├── Task3_ModelComparison.ipynb# Task 3 notebook (model comparison)
├── uber_clean.csv            # Cleaned dataset
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Pipeline Details
The app applies the **exact same data pipeline** used during training:
1. Converts user-input coordinates (degrees) to radians
2. Computes derived features: `distance` (Haversine), `bearing`, `nyc_dist`, `hour_sin/cos`
3. Feeds a 17-feature DataFrame through the sklearn pipeline (imputation → scaling → encoding → feature selection → prediction)

This ensures predictions are consistent with training — raw input never reaches the model directly.
