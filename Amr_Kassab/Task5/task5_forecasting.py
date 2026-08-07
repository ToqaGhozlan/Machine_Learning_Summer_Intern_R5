import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('plots', exist_ok=True)

df = pd.read_csv('cleaned_weather_hourly.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime').sort_index()
df = df.asfreq('h')

df['temperature'] = df['temperature'].interpolate(method='time', limit_direction='both')
df['relative_humidity'] = df['relative_humidity'].interpolate(method='linear', limit_direction='both')
df['wind_speed'] = df['wind_speed'].interpolate(method='linear', limit_direction='both')
df['precipitation'] = df['precipitation'].fillna(0.0)

roll_mean_24_temp = df['temperature'].rolling(window=24, min_periods=1).mean()
roll_std_24_temp = df['temperature'].rolling(window=24, min_periods=1).std().fillna(1.0)
z_score_temp = (df['temperature'] - roll_mean_24_temp) / roll_std_24_temp
upper_temp = roll_mean_24_temp + 3 * roll_std_24_temp
lower_temp = roll_mean_24_temp - 3 * roll_std_24_temp
df['temperature'] = np.where(z_score_temp > 3, upper_temp, np.where(z_score_temp < -3, lower_temp, df['temperature']))

df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['day'] = df.index.day
df['month'] = df.index.month
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12.0)

df['lag_1'] = df['temperature'].shift(1)
df['lag_24'] = df['temperature'].shift(24)
df['lag_168'] = df['temperature'].shift(168)

df['roll_mean_24'] = df['temperature'].rolling(24).mean()
df['roll_std_24'] = df['temperature'].rolling(24).std()
df['roll_mean_168'] = df['temperature'].rolling(168).mean()
df['roll_std_168'] = df['temperature'].rolling(168).std()

test_hours = 504
train_df = df.iloc[:-test_hours]
test_df = df.iloc[-test_hours:]

adf_raw = adfuller(train_df['temperature'].iloc[-4000:].dropna())
kpss_raw = kpss(train_df['temperature'].iloc[-4000:].dropna(), regression='c')

diff_1 = train_df['temperature'].diff(1).dropna()
diff_24 = train_df['temperature'].diff(24).dropna()

adf_diff1 = adfuller(diff_1.iloc[-4000:])
kpss_diff1 = kpss(diff_1.iloc[-4000:], regression='c')

adf_diff24 = adfuller(diff_24.iloc[-4000:])
kpss_diff24 = kpss(diff_24.iloc[-4000:], regression='c')

print("Step 1: Data loaded and preprocessed.", flush=True)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
plot_acf(train_df['temperature'].iloc[-2000:].dropna(), lags=48, ax=axes[0, 0], title='ACF - Raw Temperature')
plot_pacf(train_df['temperature'].iloc[-2000:].dropna(), lags=48, ax=axes[0, 1], title='PACF - Raw Temperature')
plot_acf(diff_1.iloc[-2000:], lags=48, ax=axes[1, 0], title='ACF - 1st Difference (d=1)')
plot_pacf(diff_1.iloc[-2000:], lags=48, ax=axes[1, 1], title='PACF - 1st Difference (d=1)')
plot_acf(diff_24.iloc[-2000:], lags=48, ax=axes[2, 0], title='ACF - Seasonal Difference (D=1, m=24)')
plot_pacf(diff_24.iloc[-2000:], lags=48, ax=axes[2, 1], title='PACF - Seasonal Difference (D=1, m=24)')
plt.tight_layout()
plt.savefig('plots/acf_pacf.png', dpi=300)
plt.savefig('plots/sarima_acf_pacf.png', dpi=300)
plt.close()

print("Step 2: Running auto_arima...", flush=True)
auto_model = pm.auto_arima(
    train_df['temperature'].iloc[-500:],
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    max_p=2,
    max_q=2,
    d=1,
    maxiter=10
)

print("Step 3: Fitting SARIMA model...", flush=True)
sarima_order = (2, 1, 2)
sarima_seasonal_order = (1, 1, 1, 24)

sarima_model = SARIMAX(
    train_df['temperature'].iloc[-2000:],
    order=sarima_order,
    seasonal_order=sarima_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fit = sarima_model.fit(disp=False, maxiter=20)
print("Step 4: SARIMA model fitted successfully.")

fig = sarima_fit.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.savefig('plots/sarima_diagnostics.png', dpi=300)
plt.close()

lb_res = acorr_ljungbox(sarima_fit.resid.dropna(), lags=[24], return_df=True)

sarima_forecast = sarima_fit.get_forecast(steps=len(test_df))
sarima_pred = sarima_forecast.predicted_mean
sarima_ci = sarima_forecast.conf_int(alpha=0.05)

plt.figure(figsize=(14, 5))
plt.plot(train_df.index[-200:], train_df['temperature'][-200:], label='Train (Last 200h)', color='#1f77b4')
plt.plot(test_df.index, test_df['temperature'], label='Actual Test Data', color='#2ca02c')
plt.plot(test_df.index, sarima_pred, label='SARIMA Forecast', color='crimson')
plt.fill_between(test_df.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='crimson', alpha=0.2, label='95% Confidence Interval')
plt.title('SARIMA Temperature Forecast vs Actual (21-Day Horizon)', fontweight='bold', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('plots/sarima_forecast.png', dpi=300)
plt.close()

naive_pred = test_df['lag_24'].fillna(train_df['temperature'].iloc[-24])

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'sMAPE': smape}

naive_metrics = compute_metrics(test_df['temperature'], naive_pred)
sarima_metrics = compute_metrics(test_df['temperature'], sarima_pred)

tscv = TimeSeriesSplit(n_splits=5, test_size=24)
wf_scores = []
for tr_idx, te_idx in tscv.split(df['temperature']):
    tr_series, te_series = df['temperature'].iloc[tr_idx], df['temperature'].iloc[te_idx]
    m_wf = SARIMAX(tr_series.iloc[-1000:], order=(1, 1, 1), seasonal_order=(1, 1, 0, 24), enforce_stationarity=False, enforce_invertibility=False)
    fit_wf = m_wf.fit(disp=False, maxiter=10)
    pred_wf = fit_wf.get_forecast(steps=len(te_series)).predicted_mean
    wf_scores.append(mean_absolute_error(te_series, pred_wf))
mean_wf_mae = np.mean(wf_scores)

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df[['temperature']].values)
test_scaled = scaler.transform(test_df[['temperature']].values)

full_scaled = np.vstack((train_scaled, test_scaled))

def create_window_sequences(series, n_steps=48):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 48
torch.set_num_threads(4)
torch.manual_seed(42)
np.random.seed(42)

train_scaled_recent = train_scaled[-4000:]
X_train, y_train = create_window_sequences(train_scaled_recent, n_steps=n_steps)

test_combined = np.vstack((train_scaled[-n_steps:], test_scaled))
X_test, y_test = create_window_sequences(test_combined, n_steps=n_steps)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

val_size = int(len(X_train_t) * 0.1)
X_tr, y_tr = X_train_t[:-val_size], y_train_t[:-val_size]
X_val, y_val = X_train_t[-val_size:], y_train_t[-val_size:]

rnn_net = RNNModel(hidden_size=32)
optimizer_rnn = optim.Adam(rnn_net.parameters(), lr=0.005)
criterion = nn.MSELoss()

dataset_tr = torch.utils.data.TensorDataset(X_tr, y_tr)
loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=256, shuffle=True)

for epoch in range(15):
    rnn_net.train()
    for bx, by in loader_tr:
        optimizer_rnn.zero_grad()
        out = rnn_net(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer_rnn.step()

rnn_net.eval()
with torch.no_grad():
    rnn_pred_scaled = rnn_net(X_test_t).numpy()

rnn_pred = scaler.inverse_transform(rnn_pred_scaled)
rnn_metrics = compute_metrics(test_df['temperature'].values, rnn_pred.flatten())

lstm_net = LSTMModel(hidden_size=64, num_layers=2, dropout=0.2)
optimizer_lstm = optim.Adam(lstm_net.parameters(), lr=0.003)

best_val_loss = float('inf')
patience, counter = 5, 0
best_model_weights = None
train_losses, val_losses = [], []

for epoch in range(25):
    lstm_net.train()
    running_loss = 0.0
    for bx, by in loader_tr:
        optimizer_lstm.zero_grad()
        out = lstm_net(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer_lstm.step()
        running_loss += loss.item() * len(bx)
    
    epoch_train_loss = running_loss / len(X_tr)
    
    lstm_net.eval()
    with torch.no_grad():
        val_out = lstm_net(X_val)
        epoch_val_loss = criterion(val_out, y_val).item()
    
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_weights = lstm_net.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            break

if best_model_weights is not None:
    lstm_net.load_state_dict(best_model_weights)

lstm_net.eval()
with torch.no_grad():
    lstm_pred_scaled = lstm_net(X_test_t).numpy()

lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
lstm_metrics = compute_metrics(test_df['temperature'].values, lstm_pred.flatten())

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)
plt.title('LSTM Training and Validation Loss Curves', fontweight='bold', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.savefig('plots/lstm_loss.png', dpi=300)
plt.close()

plt.figure(figsize=(14, 5))
plt.plot(test_df.index, test_df['temperature'], label='Actual Temperature', color='black', alpha=0.8, linewidth=1.5)
plt.plot(test_df.index, sarima_pred, label=f"SARIMA (MAE: {sarima_metrics['MAE']:.2f}°C)", color='#d62728', linestyle='--')
plt.plot(test_df.index, rnn_pred, label=f"Simple RNN (MAE: {rnn_metrics['MAE']:.2f}°C)", color='#2ca02c', alpha=0.7)
plt.plot(test_df.index, lstm_pred, label=f"LSTM (MAE: {lstm_metrics['MAE']:.2f}°C)", color='#1f77b4', linewidth=1.8)
plt.title('Model Comparison: SARIMA vs Simple RNN vs LSTM', fontweight='bold', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=300)
plt.close()

models = ['Seasonal Naive', 'SARIMA (2,1,2)(1,1,1)24', 'Simple RNN', 'LSTM']
maes = [naive_metrics['MAE'], sarima_metrics['MAE'], rnn_metrics['MAE'], lstm_metrics['MAE']]
rmses = [naive_metrics['RMSE'], sarima_metrics['RMSE'], rnn_metrics['RMSE'], lstm_metrics['RMSE']]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, maes, width, label='MAE (°C)', color='#4c72b0')
rects2 = ax.bar(x + width/2, rmses, width, label='RMSE (°C)', color='#dd8452')

ax.set_ylabel('Error (°C)', fontweight='bold')
ax.set_title('Forecast Error Metrics Across Models', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('plots/model_comparison_bar.png', dpi=300)
plt.close()

horizons = list(range(1, 25))
sarima_horizon_errors = []
lstm_horizon_errors = []

for h in horizons:
    s_err = np.abs(test_df['temperature'].values[:h] - sarima_pred.values[:h]).mean()
    l_err = np.abs(test_df['temperature'].values[:h] - lstm_pred.flatten()[:h]).mean()
    sarima_horizon_errors.append(s_err)
    lstm_horizon_errors.append(l_err)

plt.figure(figsize=(10, 5))
plt.plot(horizons, sarima_horizon_errors, marker='o', label='SARIMA Error Growth', color='#d62728')
plt.plot(horizons, lstm_horizon_errors, marker='s', label='LSTM Error Growth', color='#1f77b4')
plt.title('Error Growth by Forecast Horizon (t+1 to t+24)', fontweight='bold', fontsize=14)
plt.xlabel('Forecast Horizon (Hours ahead)')
plt.ylabel('Cumulative Mean Absolute Error (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/error_by_horizon.png', dpi=300)
plt.close()

results_summary = {
    'stationarity': {
        'adf_raw_pvalue': float(adf_raw[1]),
        'kpss_raw_pvalue': float(kpss_raw[1]),
        'adf_diff1_pvalue': float(adf_diff1[1]),
        'kpss_diff1_pvalue': float(kpss_diff1[1]),
        'adf_diff24_pvalue': float(adf_diff24[1]),
        'kpss_diff24_pvalue': float(kpss_diff24[1])
    },
    'auto_arima_order': str(auto_model.order),
    'auto_arima_seasonal_order': str(auto_model.seasonal_order),
    'ljung_box_pvalue': float(lb_res['lb_pvalue'].iloc[0]),
    'walk_forward_mae': float(mean_wf_mae),
    'metrics': {
        'Naive': naive_metrics,
        'SARIMA': sarima_metrics,
        'RNN': rnn_metrics,
        'LSTM': lstm_metrics
    }
}

with open('task5_results.json', 'w') as f:
    json.dump(results_summary, f, indent=4)

print('Task 5 execution complete. All outputs generated.')
