import os
import json
import joblib
import numpy as np
import tensorflow as tf

class WeatherPipeline:
    """
    A single pipeline class that encapsulates both the data scaler and the ML model.
    It loads them once upon initialization.
    """
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'ml_models')
        
        self.scaler_path = os.path.join(models_dir, 'scaler.pkl')
        self.model_path = os.path.join(models_dir, 'simple_rnn_model.keras')
        self.sample_data_path = os.path.join(models_dir, 'sample_data.json')
        
        self.scaler = None
        self.model = None
        self.feature_names = ['temp_mean', 'humidity', 'wind_speed', 'precipitation']
        
        self.load_pipeline()

    def load_pipeline(self):
        # Load the scaler
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        
        # Load the model
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)

    def predict(self, feature_dict):
        """
        Accepts a dictionary of features and returns the predicted temp_mean.
        """
        if self.scaler is None or self.model is None:
            raise RuntimeError("Pipeline components (scaler or model) are not loaded.")

        # Extract features in the correct order
        try:
            input_list = [feature_dict[feat] for feat in self.feature_names]
        except KeyError as e:
            raise ValueError(f"Missing required feature: {e}")

        # Convert to numpy array and shape properly for scaler (1, 4)
        data = np.array(input_list).reshape(1, 4)
        
        # 1. Scale data
        scaled_data = self.scaler.transform(data)
        
        # 2. Reshape for RNN (batch_size=1, time_steps=1, features=4)
        X = scaled_data.reshape(1, 1, 4)
        
        # 3. Predict
        scaled_pred = self.model.predict(X, verbose=0)
        
        # 4. Inverse transform
        # Create a dummy array with 4 features to inverse transform
        dummy = np.zeros((1, 4))
        dummy[0, 0] = scaled_pred[0][0]
        pred = self.scaler.inverse_transform(dummy)[0, 0]
        
        return float(pred)
        
    def get_default_data(self):
        if os.path.exists(self.sample_data_path):
            with open(self.sample_data_path, 'r') as f:
                return json.load(f)
        return {}

# Singleton instance to be used across the app
pipeline = WeatherPipeline()
