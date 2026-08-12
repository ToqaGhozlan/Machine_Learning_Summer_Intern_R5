import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler

def make_windows(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps, :])
        # Target is next day's temp_mean (index 0)
        y.append(data[i+n_steps, 0])
    return np.array(X), np.array(y)

def main():
    print("Loading data...")
    df = pd.read_csv('../task5/cairo_weather_clean.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Preprocessing
    df = df.infer_objects(copy=False)
    df.interpolate(method='time', limit_direction='both', inplace=True)
    roll_mean = df['temp_mean'].rolling(30, center=True).mean().bfill().ffill()
    roll_std = df['temp_mean'].rolling(30, center=True).std().bfill().ffill()
    df['temp_mean'] = df['temp_mean'].clip(lower=roll_mean - 3*roll_std, upper=roll_mean + 3*roll_std)
    
    # Define features
    features = ['temp_mean', 'humidity', 'wind_speed', 'precipitation']
    print("Preparing data...")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    n_steps = 1
    X, y = make_windows(scaled_data, n_steps)
    
    print(f"Training Simple RNN on {X.shape[0]} samples with shape {X.shape}...")
    tf.random.set_seed(42)
    
    model = Sequential([
        SimpleRNN(32, input_shape=(n_steps, len(features))),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=32, verbose=1)
    
    print("Creating output directory...")
    out_dir = os.path.join('weather_project', 'predictor', 'ml_models')
    os.makedirs(out_dir, exist_ok=True)
    
    model_path = os.path.join(out_dir, 'simple_rnn_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    scaler_path = os.path.join(out_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save a single sample (last day in dataset) for pre-filling the form
    sample_data = df[features].iloc[-1].to_dict()
    sample_data = {k: round(float(v), 2) for k, v in sample_data.items()}
    json_path = os.path.join(out_dir, 'sample_data.json')
    with open(json_path, 'w') as f:
        json.dump(sample_data, f)
    print(f"Sample data saved to {json_path}")
    
    print("Training and export complete!")

if __name__ == "__main__":
    main()
