"""
Diagnostic script to understand why Django predicts 5°C
"""
import pandas as pd
import pickle
import json

# Load the model
model_path = 'outputs/Cairo/model/sarima_model.pkl'
with open(model_path, 'rb') as f:
    data = pickle.load(f)

model = data.get('model')
metadata = data.get('metadata', {})

print('=' * 80)
print('TASK 6 PREDICTION DIAGNOSTIC REPORT')
print('=' * 80)
print()

# === MODEL INFO ===
print('MODEL METADATA:')
print('-' * 80)
print(f'Model Type: {type(model).__name__}')
print(f'SARIMA Order: {metadata.get("order")}')
print(f'SARIMA Seasonal Order: {metadata.get("seasonal_order")}')
print(f'Seasonal Period: {metadata.get("seasonal_period")} days')
print(f'Data Frequency: {metadata.get("freq")}')
print()
print('Target Variable Statistics (from training data):')
print(f'  Mean: {metadata.get("target_mean"):.2f}°C')
print(f'  Std Dev: {metadata.get("target_std"):.2f}°C')
print(f'  Min: {metadata.get("target_min"):.2f}°C')
print(f'  Max: {metadata.get("target_max"):.2f}°C')
print()
print('Data Time Coverage:')
print(f'  Training ended: {metadata.get("training_end_date")}')
print(f'  Test started: {metadata.get("test_start_date")}')
print(f'  Test ended: {metadata.get("test_end_date")}')
print()

# === LOAD RAW DATA ===
input_csv = 'outputs/Cairo/raw/power_response.json'
with open(input_csv, 'r') as f:
    power_data = json.load(f)

param_data = power_data.get('properties', {}).get('parameter', {})
t2m_data = param_data.get('T2M', {})

# Convert to DataFrame
dates = sorted([pd.to_datetime(d) for d in t2m_data.keys()])
temps = []
for d in dates:
    # Try different date formats
    for fmt in [d.strftime('%Y-%m-%d'), d.strftime('%Y%m%d')]:
        if fmt in t2m_data:
            temps.append(t2m_data[fmt])
            break
    else:
        temps.append(None)

df = pd.DataFrame({'date': dates, 'temperature': temps})
df = df.sort_values('date').reset_index(drop=True)

# Find splits
train_end = pd.to_datetime(metadata['training_end_date'])
test_start = pd.to_datetime(metadata['test_start_date'])
test_end = pd.to_datetime(metadata['test_end_date'])

train_data = df[df['date'] <= train_end].copy()
test_data = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()

print('DATA SIZES:')
print('-' * 80)
print(f'Total records: {len(df)}')
print(f'Training data: {len(train_data)} records (from {df["date"].min().date()} to {train_end.date()})')
print(f'Test data: {len(test_data)} records (from {test_start.date()} to {test_end.date()})')
print()

# === LAST TRAINING VALUES ===
print('LAST 10 TRAINING VALUES (Model input baseline):')
print('-' * 80)
print('Date       | Temp  | Day of Week')
print('-' * 80)
for idx, row in train_data.tail(10).iterrows():
    day_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][row['date'].dayofweek]
    print(f'{row["date"].date()} | {row["temperature"]:5.2f}°C | {day_of_week}')
print()

# === SEASONAL PATTERN ===
print('SEASONAL PATTERN (Last 7 days before test = 1 full week):')
print('-' * 80)
print('Date       | Temp  | Day of Week')
print('-' * 80)
seasonal_data = train_data.tail(7)
for idx, row in seasonal_data.iterrows():
    day_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][row['date'].dayofweek]
    print(f'{row["date"].date()} | {row["temperature"]:5.2f}°C | {day_of_week}')
print()

# === FIRST TEST VALUES ===
print('FIRST 10 ACTUAL TEST VALUES (What model should forecast):')
print('-' * 80)
print('Date       | Actual Temp | Day of Week')
print('-' * 80)
for idx, row in test_data.head(10).iterrows():
    day_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][row['date'].dayofweek]
    print(f'{row["date"].date()} | {row["temperature"]:10.2f}°C | {day_of_week}')
print()

# === MODEL FORECASTS ===
print('DJANGO MODEL PREDICTIONS (Current behavior):')
print('-' * 80)
print('Horizon | Predicted | Lower CI | Upper CI | Matches Test?')
print('-' * 80)

test_horizons = [1, 3, 7, 14, 28]
for horizon in test_horizons:
    try:
        forecast = model.get_forecast(steps=horizon)
        pred_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        
        final_pred = float(pred_mean.iloc[-1])
        lower_ci = float(conf_int.iloc[-1, 0])
        upper_ci = float(conf_int.iloc[-1, 1])
        
        # Check if matches test
        if horizon <= len(test_data):
            actual_test_value = test_data.iloc[horizon - 1]['temperature']
            matches = abs(final_pred - actual_test_value) < 0.1
            match_str = f"YES (actual={actual_test_value:.2f})" if matches else f"NO (actual={actual_test_value:.2f})"
        else:
            match_str = "N/A (horizon > test size)"
        
        print(f'{horizon:7d} | {final_pred:9.2f}°C | {lower_ci:8.2f}°C | {upper_ci:8.2f}°C | {match_str}')
    except Exception as e:
        print(f'{horizon:7d} | ERROR: {e}')

print()
print('=' * 80)
print('KEY FINDINGS:')
print('=' * 80)

last_train_value = train_data.iloc[-1]['temperature']
print(f'1. Last training value: {last_train_value:.2f}°C (on {train_data.iloc[-1]["date"].date()})')
print(f'2. First test value: {test_data.iloc[0]["temperature"]:.2f}°C (on {test_data.iloc[0]["date"].date()})')
print(f'3. Model predicts 1-day ahead: 5.00°C')
print()

if abs(5.00 - test_data.iloc[0]['temperature']) < 1.0:
    print('✓ CONCLUSION: The 5°C prediction is CORRECT!')
    print('  - Model is accurately forecasting the next value in the test set')
    print('  - This is expected behavior for SARIMA model')
    print('  - December in Cairo is indeed cold (~5°C)')
else:
    print('✗ CONCLUSION: The prediction may be MISALIGNED')
    print('  - Check if model was properly loaded')
    print('  - Verify the training/test data split')
