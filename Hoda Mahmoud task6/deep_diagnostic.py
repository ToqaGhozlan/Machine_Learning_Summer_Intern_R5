"""
Deep dive into model state and why forecast is misaligned
"""
import pandas as pd
import pickle
import json
import numpy as np

# Load the model
model_path = 'outputs/Cairo/model/sarima_model.pkl'
with open(model_path, 'rb') as f:
    data = pickle.load(f)

model = data.get('model')
metadata = data.get('metadata', {})

print('=' * 90)
print('DEEP DIAGNOSTIC: Model Internal State Analysis')
print('=' * 90)
print()

# === MODEL FITTED DATA ===
print('MODEL FITTED DATA STATE:')
print('-' * 90)
print(f'Model type: {type(model).__name__}')
print(f'Model endog (observations used): {len(model.model.endog)} values')
try:
    endog_array = np.asarray(model.model.endog).flatten()
    print(f'Last endog value (what model sees): {float(endog_array[-1]):.4f}')
except:
    print(f'Last endog value (what model sees): Could not extract')
print(f'Expected last value: {metadata["target_max"]:.2f}°C (from metadata)')
print()

# Check the model's fittedvalues
print(f'Fitted values shape: {model.fittedvalues.shape}')
print(f'Last 5 fitted values:')
for i, val in enumerate(model.fittedvalues[-5:]):
    print(f'  {i}: {float(val):.4f}')
print()

# === LOAD ACTUAL TRAINING DATA ===
input_csv = 'outputs/Cairo/raw/power_response.json'
with open(input_csv, 'r') as f:
    power_data = json.load(f)

param_data = power_data.get('properties', {}).get('parameter', {})
t2m_data = param_data.get('T2M', {})

dates = sorted([pd.to_datetime(d) for d in t2m_data.keys()])
temps = []
for d in dates:
    for fmt in [d.strftime('%Y-%m-%d'), d.strftime('%Y%m%d')]:
        if fmt in t2m_data:
            temps.append(t2m_data[fmt])
            break
    else:
        temps.append(None)

df = pd.DataFrame({'date': dates, 'temperature': temps})
df = df.sort_values('date').reset_index(drop=True)

train_end = pd.to_datetime(metadata['training_end_date'])
train_data = df[df['date'] <= train_end].copy()

print('ACTUAL TRAINING DATA (what Performance.py used):')
print('-' * 90)
print(f'Training data size: {len(train_data)}')
print(f'Last 5 values in training data:')
for idx, row in train_data.tail(5).iterrows():
    print(f'  {row["date"].date()}: {row["temperature"]:.4f}')
print()
print(f'Training data series min: {train_data["temperature"].min():.4f}')
print(f'Training data series max: {train_data["temperature"].max():.4f}')
print(f'Training data series mean: {train_data["temperature"].mean():.4f}')
print()

# === CHECK MODEL FORECAST STRUCTURE ===
print('MODEL FORECAST STRUCTURE ANALYSIS:')
print('-' * 90)

# Get forecast and inspect
forecast = model.get_forecast(steps=7)
pred_mean = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)

print(f'Forecast steps: {len(pred_mean)}')
print(f'Predicted mean type: {type(pred_mean)}')
print(f'Predicted mean index: {pred_mean.index[0]} to {pred_mean.index[-1]}')
print()
print('7-day forecast from model:')
for i, (date, pred) in enumerate(pred_mean.items()):
    lower = conf_int.iloc[i, 0]
    upper = conf_int.iloc[i, 1]
    print(f'  {i+1}: {date.date()} -> {float(pred):6.2f}°C [{float(lower):6.2f}, {float(upper):6.2f}]')

print()

# === ISSUE ANALYSIS ===
print('=' * 90)
print('ISSUE DIAGNOSIS:')
print('=' * 90)
print()

last_train_val = train_data.iloc[-1]['temperature']
first_pred = float(pred_mean.iloc[0])

print(f'Last training value: {last_train_val:.2f}°C')
print(f'First forecast (1-day ahead): {first_pred:.2f}°C')
print(f'Difference: {abs(first_pred - last_train_val):.2f}°C')
print()

# Check mean value
train_mean = train_data['temperature'].mean()
print(f'Training data mean: {train_mean:.2f}°C')
print(f'Forecast seems to revert to: ~{first_pred:.2f}°C')
print()

if abs(first_pred - train_mean) < 1.0:
    print('⚠ PROBLEM FOUND: Model is predicting the MEAN, not continuing the time series!')
    print('   This suggests the model was fitted INCORRECTLY or with DIFFERENT DATA')
elif abs(first_pred - last_train_val) < 0.5:
    print('✓ Model correctly continues from last value')
else:
    print('⚠ Model is applying trend/seasonal adjustment')
    print('   But mismatch from actual test values suggests data mismatch')

print()
print('=' * 90)
print('HYPOTHESIS: Data Used to Train Model vs Data Provided Now')
print('=' * 90)
print()
print('The model in the pickle file was fitted on DIFFERENT data than what we have now!')
print('This could happen if:')
print('  1. Performance.py was run, data changed, then run again')
print('  2. The JSON data was updated between runs')
print('  3. There was an issue with preprocessing/imputation when the model was trained')
print()
print('SOLUTION: Re-run Performance.py to retrain the model with current data!')
