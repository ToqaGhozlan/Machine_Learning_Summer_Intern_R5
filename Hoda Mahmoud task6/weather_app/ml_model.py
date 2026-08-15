"""
ML Model loading and inference module for SARIMA weather predictions.
This module loads the trained SARIMA model and provides prediction functionality.
"""

import os
import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Global model instance - loaded once at startup
_model_instance = None
_model_metadata = None


def load_model():
    """
    Load the trained SARIMA model from disk.
    This is called once when the Django app starts.
    
    Returns:
        tuple: (model, metadata) or (None, None) if model cannot be loaded
    """
    global _model_instance, _model_metadata
    
    if _model_instance is not None:
        return _model_instance, _model_metadata
    
    model_path = settings.ML_MODEL_PATH
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None, None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            _model_instance = model_data.get('model')
            _model_metadata = model_data.get('metadata', {})
        logger.info(f"SARIMA model loaded successfully from {model_path}")
        return _model_instance, _model_metadata
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None, None


def get_model():
    """
    Get the loaded model instance. If not loaded, attempt to load it.
    
    Returns:
        tuple: (model, metadata) or (None, None) if model is not available
    """
    if _model_instance is None:
        return load_model()
    return _model_instance, _model_metadata


def predict_temperature(horizon_days=1):
    """
    Predict temperature for the specified number of days ahead.
    
    Args:
        horizon_days (int): Number of days to forecast ahead (1-28)
    
    Returns:
        dict: Prediction results with keys:
            - 'success' (bool): Whether prediction was successful
            - 'prediction' (float): Predicted temperature value(s) or None
            - 'error' (str): Error message if unsuccessful
            - 'metadata' (dict): Model metadata
            - 'input_horizon' (int): The requested forecast horizon
    """
    
    # Validate input
    if not isinstance(horizon_days, int) or horizon_days < 1 or horizon_days > 28:
        return {
            'success': False,
            'prediction': None,
            'error': 'Forecast horizon must be between 1 and 28 days',
            'metadata': None,
            'input_horizon': horizon_days
        }
    
    model, metadata = get_model()
    
    if model is None:
        return {
            'success': False,
            'prediction': None,
            'error': 'Model is not available. Please check model file.',
            'metadata': None,
            'input_horizon': horizon_days
        }
    
    try:
        # Get forecast
        forecast = model.get_forecast(steps=horizon_days)
        predicted_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        
        # Return the final prediction (at the horizon)
        final_prediction = float(predicted_mean.iloc[-1])
        lower_ci = float(conf_int.iloc[-1, 0])
        upper_ci = float(conf_int.iloc[-1, 1])
        
        return {
            'success': True,
            'prediction': final_prediction,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'horizon_days': horizon_days,
            'error': None,
            'metadata': metadata
        }
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return {
            'success': False,
            'prediction': None,
            'error': f'Prediction error: {str(e)}',
            'metadata': metadata,
            'input_horizon': horizon_days
        }


def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        dict: Model information or error dictionary
    """
    model, metadata = get_model()
    
    if model is None:
        return {
            'status': 'error',
            'message': 'Model is not available'
        }
    
    return {
        'status': 'loaded',
        'order': metadata.get('order'),
        'seasonal_order': metadata.get('seasonal_order'),
        'seasonal_period': metadata.get('seasonal_period'),
        'frequency': metadata.get('freq'),
        'target_mean': metadata.get('target_mean'),
        'target_std': metadata.get('target_std'),
        'target_min': metadata.get('target_min'),
        'target_max': metadata.get('target_max'),
        'training_end_date': metadata.get('training_end_date'),
        'test_start_date': metadata.get('test_start_date'),
        'test_end_date': metadata.get('test_end_date'),
    }
