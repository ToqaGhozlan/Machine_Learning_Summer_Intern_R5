"""
Train and save weather prediction models using the cleaned hourly weather dataset
from Task 4/5. Trains a Gradient Boosting Regressor and a Random Forest as an
alternative model. Both are saved as joblib files for the Django application.

Models predict hourly temperature (°C) given:
  - relative_humidity (%)
  - precipitation (mm)
  - wind_speed (km/h)
  - cloud_cover (%)
  - surface_pressure (hPa)
  - hour (0-23)
  - month (1-12)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

TASK5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Task5')
DATA_PATH = os.path.join(TASK5_DIR, 'cleaned_weather_hourly.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictor', 'ml_models')

FEATURE_COLUMNS = [
    'relative_humidity',
    'precipitation',
    'wind_speed',
    'cloud_cover',
    'surface_pressure',
    'hour',
    'month',
    'hour_sin',
    'hour_cos',
    'month_sin',
    'month_cos',
]

TARGET_COLUMN = 'temperature'


def load_and_prepare_data():
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    df = df.asfreq('h')

    df['temperature'] = df['temperature'].interpolate(method='time', limit_direction='both')
    df['relative_humidity'] = df['relative_humidity'].interpolate(method='linear', limit_direction='both')
    df['wind_speed'] = df['wind_speed'].interpolate(method='linear', limit_direction='both')
    df['precipitation'] = df['precipitation'].fillna(0.0)
    df['cloud_cover'] = df['cloud_cover'].interpolate(method='linear', limit_direction='both')
    df['surface_pressure'] = df['surface_pressure'].interpolate(method='linear', limit_direction='both')

    roll_mean = df['temperature'].rolling(window=24, min_periods=1).mean()
    roll_std = df['temperature'].rolling(window=24, min_periods=1).std().fillna(1.0)
    z_score = (df['temperature'] - roll_mean) / roll_std
    upper = roll_mean + 3 * roll_std
    lower = roll_mean - 3 * roll_std
    df['temperature'] = np.where(z_score > 3, upper, np.where(z_score < -3, lower, df['temperature']))

    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12.0)

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    print(f"Dataset shape after preprocessing: {df.shape}")
    return df


def train_models(df):
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Training Gradient Boosting Regressor ---")
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    gb_metrics = evaluate_model(y_test, gb_pred, "Gradient Boosting")

    print("\n--- Training Random Forest Regressor ---")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")

    return gb_model, rf_model, scaler, gb_metrics, rf_metrics


def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"  MAE:  {mae:.4f} °C")
    print(f"  RMSE: {rmse:.4f} °C")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return {
        'model': model_name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2),
    }


def save_artifacts(gb_model, rf_model, scaler, gb_metrics, rf_metrics, df):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(gb_model, os.path.join(MODEL_DIR, 'gradient_boosting_model.joblib'))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, 'random_forest_model.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'feature_scaler.joblib'))

    feature_ranges = {}
    for col in ['relative_humidity', 'precipitation', 'wind_speed', 'cloud_cover', 'surface_pressure', 'temperature']:
        feature_ranges[col] = {
            'min': round(float(df[col].min()), 2),
            'max': round(float(df[col].max()), 2),
            'mean': round(float(df[col].mean()), 2),
            'std': round(float(df[col].std()), 2),
        }

    metadata = {
        'feature_columns': FEATURE_COLUMNS,
        'target_column': TARGET_COLUMN,
        'feature_ranges': feature_ranges,
        'dataset_info': {
            'source': 'Open-Meteo Historical Weather API',
            'location': 'Cairo, Egypt (30.0444°N, 31.2357°E)',
            'period': '2022-01-01 to 2023-12-31',
            'frequency': 'Hourly',
            'total_records': int(len(df)),
        },
        'models': {
            'gradient_boosting': gb_metrics,
            'random_forest': rf_metrics,
        },
    }

    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\nAll artifacts saved to: {MODEL_DIR}")
    print(f"  - gradient_boosting_model.joblib")
    print(f"  - random_forest_model.joblib")
    print(f"  - feature_scaler.joblib")
    print(f"  - model_metadata.json")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Could not find dataset at {DATA_PATH}")
        print("Make sure Task5/cleaned_weather_hourly.csv exists.")
        sys.exit(1)

    df = load_and_prepare_data()
    gb_model, rf_model, scaler, gb_metrics, rf_metrics = train_models(df)
    save_artifacts(gb_model, rf_model, scaler, gb_metrics, rf_metrics, df)
    print("\nModel training and export completed successfully.")


if __name__ == '__main__':
    main()
