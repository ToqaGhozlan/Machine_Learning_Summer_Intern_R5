# forecast/ml/model_loader.py

import os
import json
import joblib
from tensorflow.keras.models import load_model

# Base path: forecast/ml/models/
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_PATH = os.path.join(MODELS_DIR, "egypt_t2m_lstm.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "egypt_t2m_scaler.pkl")
CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")

# Loaded once when this module is first imported
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

LOOKBACK = config["lookback"]   # 365
TARGET = config["target"]       # "T2M"
FREQ = config["freq"]           # "D"

print(f"[model_loader] Loaded model, scaler, config. Lookback={LOOKBACK}")