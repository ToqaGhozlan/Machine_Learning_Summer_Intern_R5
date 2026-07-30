import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('plots', exist_ok=True)

latitude = 30.0444
longitude = 31.2357
start_date = '2022-01-01'
end_date = '2023-12-31'

api_url = 'https://archive-api.open-meteo.com/v1/archive'
params = {
    'latitude': latitude,
    'longitude': longitude,
    'start_date': start_date,
    'end_date': end_date,
    'daily': ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 'precipitation_sum', 'rain_sum', 'wind_speed_10m_max'],
    'hourly': ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m', 'cloud_cover', 'surface_pressure'],
    'timezone': 'Africa/Cairo'
}

response = requests.get(api_url, params=params)
raw_data = response.json()

with open('raw_weather_data.json', 'w') as f:
    json.dump(raw_data, f, indent=4)

daily_df = pd.DataFrame(raw_data['daily'])
daily_df['date'] = pd.to_datetime(daily_df['time'])
daily_df = daily_df.drop(columns=['time']).set_index('date')

daily_df = daily_df.rename(columns={
    'temperature_2m_mean': 'temperature_mean',
    'temperature_2m_max': 'temperature_max',
    'temperature_2m_min': 'temperature_min',
    'precipitation_sum': 'precipitation',
    'rain_sum': 'rain',
    'wind_speed_10m_max': 'wind_speed_max'
})

hourly_df = pd.DataFrame(raw_data['hourly'])
hourly_df['datetime'] = pd.to_datetime(hourly_df['time'])
hourly_df = hourly_df.drop(columns=['time']).set_index('datetime')

hourly_df = hourly_df.rename(columns={
    'temperature_2m': 'temperature',
    'relative_humidity_2m': 'relative_humidity',
    'precipitation': 'precipitation',
    'wind_speed_10m': 'wind_speed',
    'cloud_cover': 'cloud_cover',
    'surface_pressure': 'surface_pressure'
})

daily_expected_index = pd.date_range(start=start_date, end=end_date, freq='D')
daily_df = daily_df.reindex(daily_expected_index)
daily_df.index.name = 'date'

hourly_expected_index = pd.date_range(start=f'{start_date} 00:00', end=f'{end_date} 23:00', freq='h')
hourly_df = hourly_df.reindex(hourly_expected_index)
hourly_df.index.name = 'datetime'

daily_df = daily_df.interpolate(method='time').bfill().ffill()
hourly_df = hourly_df.interpolate(method='time').bfill().ffill()

daily_df['month'] = daily_df.index.month
daily_df['month_name'] = daily_df.index.strftime('%b')
daily_df['season'] = daily_df.index.month.map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})

cleaned_columns = ['temperature_mean', 'temperature_max', 'temperature_min', 'precipitation', 'rain', 'wind_speed_max', 'month', 'month_name', 'season']
daily_df[cleaned_columns].to_csv('cleaned_weather_data.csv')
hourly_df.to_csv('cleaned_weather_hourly.csv')

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Distribution of Meteorological Variables (Cairo, 2022-2023)', fontsize=16, fontweight='bold', y=0.98)

sns.histplot(daily_df['temperature_mean'], kde=True, ax=axes[0, 0], color='#1f77b4', bins=30)
axes[0, 0].set_title('Mean Daily Temperature (°C)', fontweight='bold')
axes[0, 0].set_xlabel('Temperature (°C)')

sns.histplot(daily_df['temperature_max'], kde=True, ax=axes[0, 1], color='#ff7f0e', bins=30)
axes[0, 1].set_title('Max Daily Temperature (°C)', fontweight='bold')
axes[0, 1].set_xlabel('Temperature (°C)')

sns.histplot(daily_df['temperature_min'], kde=True, ax=axes[0, 2], color='#2ca02c', bins=30)
axes[0, 2].set_title('Min Daily Temperature (°C)', fontweight='bold')
axes[0, 2].set_xlabel('Temperature (°C)')

sns.histplot(daily_df['precipitation'], kde=True, ax=axes[1, 0], color='#9467bd', bins=30)
axes[1, 0].set_title('Daily Precipitation (mm)', fontweight='bold')
axes[1, 0].set_xlabel('Precipitation (mm)')

sns.histplot(daily_df['wind_speed_max'], kde=True, ax=axes[1, 1], color='#d62728', bins=30)
axes[1, 1].set_title('Max Daily Wind Speed (km/h)', fontweight='bold')
axes[1, 1].set_xlabel('Wind Speed (km/h)')

sns.histplot(hourly_df['relative_humidity'], kde=True, ax=axes[1, 2], color='#8c564b', bins=30)
axes[1, 2].set_title('Hourly Relative Humidity (%)', fontweight='bold')
axes[1, 2].set_xlabel('Humidity (%)')

plt.tight_layout()
plt.savefig('plots/distributions.png', dpi=300)
plt.close()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
fig.suptitle('Time-Series Overview: Temperature & Precipitation (Cairo, 2022-2023)', fontsize=16, fontweight='bold')

ax1.plot(daily_df.index, daily_df['temperature_mean'], label='Mean Temp (°C)', color='#2b5c8f', linewidth=1.5)
ax1.fill_between(daily_df.index, daily_df['temperature_min'], daily_df['temperature_max'], alpha=0.2, color='#4682b4', label='Min-Max Range')
ax1.set_ylabel('Temperature (°C)', fontweight='bold')
ax1.set_title('Daily Temperature Range & Mean', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2.bar(daily_df.index, daily_df['precipitation'], color='#9467bd', width=1.0, label='Precipitation (mm)')
ax2.set_ylabel('Precipitation (mm)', fontweight='bold')
ax2.set_xlabel('Date', fontweight='bold')
ax2.set_title('Daily Precipitation Sum', fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/time_series_overview.png', dpi=300)
plt.close()

daily_df['temp_7d_mean'] = daily_df['temperature_mean'].rolling(window=7).mean()
daily_df['temp_30d_mean'] = daily_df['temperature_mean'].rolling(window=30).mean()
daily_df['temp_30d_std'] = daily_df['temperature_mean'].rolling(window=30).std()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
fig.suptitle('Rolling Statistics & Trend Analysis for Mean Temperature', fontsize=16, fontweight='bold')

ax1.plot(daily_df.index, daily_df['temperature_mean'], alpha=0.4, color='gray', label='Daily Raw Mean')
ax1.plot(daily_df.index, daily_df['temp_7d_mean'], color='#e377c2', linewidth=1.8, label='7-Day Rolling Mean')
ax1.plot(daily_df.index, daily_df['temp_30d_mean'], color='#d62728', linewidth=2.5, label='30-Day Rolling Mean')
ax1.set_ylabel('Temperature (°C)', fontweight='bold')
ax1.set_title('Long-Term Trend (7-Day vs 30-Day Rolling Averages)', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2.plot(daily_df.index, daily_df['temp_30d_std'], color='#17becf', linewidth=2.0, label='30-Day Rolling Standard Deviation')
ax2.set_ylabel('Standard Deviation (°C)', fontweight='bold')
ax2.set_xlabel('Date', fontweight='bold')
ax2.set_title('Local Volatility (30-Day Rolling Standard Deviation)', fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/rolling_statistics.png', dpi=300)
plt.close()

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Monthly Seasonality Analysis (Cairo, 2022-2023)', fontsize=16, fontweight='bold')

sns.boxplot(data=daily_df, x='month_name', y='temperature_mean', order=month_order, hue='month_name', palette='YlOrRd', ax=ax1, legend=False)
ax1.set_title('Monthly Distribution of Mean Temperature', fontweight='bold')
ax1.set_xlabel('Month', fontweight='bold')
ax1.set_ylabel('Temperature (°C)', fontweight='bold')

sns.boxplot(data=daily_df, x='month_name', y='precipitation', order=month_order, hue='month_name', palette='Blues', ax=ax2, legend=False)
ax2.set_title('Monthly Distribution of Precipitation', fontweight='bold')
ax2.set_xlabel('Month', fontweight='bold')
ax2.set_ylabel('Precipitation (mm)', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/monthly_seasonality.png', dpi=300)
plt.close()

decomp = seasonal_decompose(daily_df['temperature_mean'], model='additive', period=365)

fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
fig.suptitle('Time-Series Decomposition of Daily Mean Temperature', fontsize=16, fontweight='bold')

axes[0].plot(daily_df.index, decomp.observed, color='#1f77b4', linewidth=1.5)
axes[0].set_ylabel('Observed', fontweight='bold')

axes[1].plot(daily_df.index, decomp.trend, color='#ff7f0e', linewidth=2.0)
axes[1].set_ylabel('Trend', fontweight='bold')

axes[2].plot(daily_df.index, decomp.seasonal, color='#2ca02c', linewidth=1.5)
axes[2].set_ylabel('Seasonal', fontweight='bold')

axes[3].scatter(daily_df.index, decomp.resid, color='#d62728', s=10, alpha=0.6)
axes[3].axhline(0, color='black', linestyle='--', alpha=0.5)
axes[3].set_ylabel('Residual', fontweight='bold')
axes[3].set_xlabel('Date', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/seasonal_decomposition.png', dpi=300)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle('Autocorrelation (ACF) & Partial Autocorrelation (PACF) Analysis', fontsize=16, fontweight='bold')

plot_acf(daily_df['temperature_mean'], lags=60, ax=axes[0, 0], color='#1f77b4', title='ACF - Daily Mean Temperature')
plot_pacf(daily_df['temperature_mean'], lags=60, ax=axes[0, 1], color='#ff7f0e', title='PACF - Daily Mean Temperature')

plot_acf(daily_df['precipitation'], lags=60, ax=axes[1, 0], color='#2ca02c', title='ACF - Daily Precipitation')
plot_pacf(daily_df['precipitation'], lags=60, ax=axes[1, 1], color='#d62728', title='PACF - Daily Precipitation')

plt.tight_layout()
plt.savefig('plots/acf_pacf.png', dpi=300)
plt.close()

def run_adf_test(series, name):
    res = adfuller(series.dropna())
    print(f'=== ADF Test for {name} ===')
    print(f'ADF Statistic: {res[0]:.4f}')
    print(f'p-value: {res[1]:.4e}')
    print('Critical Values:')
    for k, v in res[4].items():
        print(f'   {k}: {v:.4f}')
    is_stat = res[1] < 0.05
    print(f'Stationary at 5% significance level: {is_stat}\n')
    return res

print('--------------------------------------------------')
print('TIME-SERIES STATIONARITY ANALYSIS (ADF TEST)')
print('--------------------------------------------------')
adf_temp = run_adf_test(daily_df['temperature_mean'], 'Daily Mean Temperature')
adf_precip = run_adf_test(daily_df['precipitation'], 'Daily Precipitation')

numeric_cols = ['temperature_mean', 'temperature_max', 'temperature_min', 'precipitation', 'rain', 'wind_speed_max']
corr_matrix = daily_df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix of Meteorological Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300)
plt.close()

rolling_window = 30
rolling_mean = daily_df['temperature_mean'].rolling(window=rolling_window).mean()
rolling_std = daily_df['temperature_mean'].rolling(window=rolling_window).std()
daily_df['z_score_temp'] = (daily_df['temperature_mean'] - rolling_mean) / rolling_std

temp_outliers = daily_df[daily_df['z_score_temp'].abs() > 3]

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(daily_df.index, daily_df['temperature_mean'], label='Mean Temp (°C)', color='#1f77b4', alpha=0.7)
ax.plot(daily_df.index, rolling_mean, label='30-Day Rolling Mean', color='black', linestyle='--')
ax.scatter(temp_outliers.index, temp_outliers['temperature_mean'], color='red', label='Outliers (|Z| > 3)', s=40, zorder=5)
ax.set_title('Outlier Detection via Rolling Z-Score (|Z| > 3)', fontsize=14, fontweight='bold')
ax.set_ylabel('Temperature (°C)', fontweight='bold')
ax.set_xlabel('Date', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/outlier_detection.png', dpi=300)
plt.close()

negative_precip_count = (daily_df['precipitation'] < 0).sum()
extreme_temp_count = ((daily_df['temperature_mean'] < -10) | (daily_df['temperature_mean'] > 60)).sum()

print('--------------------------------------------------')
print('DATA QUALITY & ANOMALY SUMMARY')
print('--------------------------------------------------')
print(f'Negative Precipitation Values: {negative_precip_count}')
print(f'Extreme Temperature Values (< -10°C or > 60°C): {extreme_temp_count}')
print(f'Total Rolling Z-Score Outliers Detected: {len(temp_outliers)}')
print('Processing completed successfully.')
