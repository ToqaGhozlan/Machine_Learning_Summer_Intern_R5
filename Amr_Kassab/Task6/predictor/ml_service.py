import os
import json
import numpy as np
import joblib
from pathlib import Path

ML_MODELS_DIR = Path(__file__).resolve().parent / 'ml_models'

_models_cache = {}


def _load_model(model_name):
    if model_name not in _models_cache:
        model_path = ML_MODELS_DIR / f'{model_name}_model.joblib'
        _models_cache[model_name] = joblib.load(model_path)
    return _models_cache[model_name]


def _load_scaler():
    if 'scaler' not in _models_cache:
        scaler_path = ML_MODELS_DIR / 'feature_scaler.joblib'
        _models_cache['scaler'] = joblib.load(scaler_path)
    return _models_cache['scaler']


def _load_metadata():
    if 'metadata' not in _models_cache:
        metadata_path = ML_MODELS_DIR / 'model_metadata.json'
        with open(metadata_path, 'r') as f:
            _models_cache['metadata'] = json.load(f)
    return _models_cache['metadata']


def get_available_models():
    return [
        ('gradient_boosting', 'Gradient Boosting Regressor'),
        ('random_forest', 'Random Forest Regressor'),
    ]


def get_feature_ranges():
    metadata = _load_metadata()
    return metadata.get('feature_ranges', {})


def get_model_metrics():
    metadata = _load_metadata()
    return metadata.get('models', {})


def get_dataset_info():
    metadata = _load_metadata()
    return metadata.get('dataset_info', {})


def predict_temperature(input_data, model_name='gradient_boosting'):
    model = _load_model(model_name)
    scaler = _load_scaler()

    hour = int(input_data['hour'])
    month = int(input_data['month'])

    features = np.array([[
        float(input_data['relative_humidity']),
        float(input_data['precipitation']),
        float(input_data['wind_speed']),
        float(input_data['cloud_cover']),
        float(input_data['surface_pressure']),
        hour,
        month,
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * month / 12.0),
        np.cos(2 * np.pi * month / 12.0),
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    metadata = _load_metadata()
    model_info = metadata.get('models', {}).get(model_name, {})
    mae = model_info.get('MAE', 2.0)
    confidence_low = round(prediction - 1.96 * mae, 1)
    confidence_high = round(prediction + 1.96 * mae, 1)

    return {
        'temperature': round(float(prediction), 1),
        'confidence_low': confidence_low,
        'confidence_high': confidence_high,
        'model_used': model_name,
        'model_mae': mae,
        'model_r2': model_info.get('R2', None),
    }
