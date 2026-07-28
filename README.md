# 🌤️ Cairo Weather Analytics & Time-Series EDA

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Time--Series-4B6584?style=for-the-badge)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A comprehensive, production-grade **Time-Series Exploratory Data Analysis (EDA)** and **Interactive Analytics Dashboard** evaluating 365 daily weather observations for **Cairo, Egypt (2025)** retrieved from the **Open-Meteo ERA5 Historical Archive API**.

---

## 📌 Executive Summary

This repository delivers a rigorous end-to-end time-series analytical workflow designed to evaluate raw climate variables, extract underlying trend and seasonal patterns, conduct formal stationarity testing, detect anomaly spikes, and deliver interactive visualization dashboards.

### 🌟 Key Findings
- **Sinusoidal Thermal Regimes**: Annual mean temperature stands at **24.2°C**, spanning from a winter low of **6.9°C** (January) to a summer peak of **44.4°C** (July/August heat waves).
- **Strong Inverse Thermo-Hydric Relationship**: Mean daily temperature and relative humidity exhibit a strong negative correlation (**r = -0.78**), indicating hyper-arid conditions during summer peaks.
- **Hyper-Arid Precipitation Profile**: Annual total rainfall reached only **14.2 mm** across 365 days, with **~350 completely dry days (0 mm)**, validating Cairo's *BWh* hyper-arid desert climate classification.
- **Non-Stationarity Validated**: Augmented Dickey-Fuller (ADF) test yielded **$p = 0.7874 > 0.05$**, confirming the raw daily temperature series is non-stationary and requires seasonal differencing ($D=1, \text{period}=365$) or STL detrending prior to time-series forecasting (SARIMA/Prophet).

---

## 📊 Interactive Dashboard Preview

The repository includes a standalone, self-contained **Interactive Web Dashboard** (`dashboard.html`) featuring **46 Plotly visualizations** structured into 6 logical analytical tabs:

1. **📊 Distributions**: Histograms and Kernel Density Estimation (KDE) for all 9 meteorological features.
2. **📈 Temporal Trends**: Daily max/min temperature, precipitation bar charts, and 7-day rolling mean smoothers.
3. **🔬 Decomposition & ACF**: STL decomposition (Trend, 30-day Seasonality, Residuals) and Autocorrelation Function (ACF / PACF) plots.
4. **🔗 Correlations**: Pairwise correlation heatmap, 4-variable scatter matrix, multi-panel dashboard, calendar heatmap, and lag plots ($k=1, k=7$).
5. **🗓️ Seasonal Analysis**: Seasonally grouped boxplots, parallel coordinates, and 3D scatter interaction space (Temperature × Humidity × Pressure).
6. **⚠️ Volatility & Outliers**: 7-day rolling standard deviation volatility tracking and local rolling Z-score outlier detection ($|Z| > 2.5$).

---

## 🛠️ Technical Workflow & Analysis Breakdown

### 1. Data Acquisition & Structuring (`Part 1`)
- **Source**: Open-Meteo Historical Weather API (`ECMWF ERA5 Reanalysis`).
- **Target Coordinates**: Cairo, Egypt (`30.0444°N, 31.2357°E`).
- **Time Horizon**: Full Year 2025 (365 daily observations).
- **Data Quality**: **100% complete** (0 missing values, 0 duplicate timestamps, clean daily frequency `freq='D'`).

### 2. Time-Series Decomposition & Stationarity (`Part 2 & 3`)
- **STL Decomposition**: Decomposed daily mean temperature with `period=30` into smooth trend, periodic seasonal, and zero-centered residual components.
- **ADF Stationarity Test**:
  - **ADF Statistic**: `-0.9019`
  - **p-value**: `0.7874`
  - **Conclusion**: Fail to reject $H_0$ — series contains a unit root.

```
ADF Test Result:
- Statistic: -0.9019
- p-value  : 0.7874 (Fail to reject H0 -> Non-Stationary)
- Critical Values: 1% (-3.449), 5% (-2.870), 10% (-2.571)
```

### 3. Anomaly Detection & Volatility (`Part 3`)
- **Rolling Z-Score Outlier Flagging**: Utilized a 30-day rolling local window to compute dynamic Z-scores, preventing seasonal bias.
- **Spring Volatility Peak**: Rolling standard deviation identified Spring (March–May) as the most volatile season due to erratic *Khamsin* heat waves and Mediterranean frontal passages.

---

## 📂 Repository Structure

```gcode
Weather Project/
├── weather_erd.ipynb        # Primary Jupyter Notebook (46 code cells + 59 markdown analyses)
├── dashboard.html           # Full Interactive Plotly Web Dashboard (Self-contained, 1.8 MB)
├── weather_cleaned.csv      # Cleaned daily weather dataset (365 rows x 11 columns)
├── weather_raw.csv          # Raw exported API dataset
├── weather.csv              # Initial exported dataset
├── requirements.txt         # Python dependency manifest
├── .gitignore               # Environment & cache exclusion rules
└── README.md                # Project documentation & executive summary
```

---

## 🚀 How to Run Locally

### Prerequisites
Ensure Python 3.11+ is installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/weather-time-series-eda.git
cd weather-time-series-eda
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch the Jupyter Notebook
```bash
jupyter notebook weather_erd.ipynb
```

### 4. View the Interactive Dashboard
You can open `dashboard.html` directly in any web browser, or host it locally:
```bash
# Host using Python's built-in HTTP server
python -m http.server 8000
```
Then navigate to `http://localhost:8000/dashboard.html` in your browser.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## ✍️ Author & Credits

- **Data Source**: [Open-Meteo ERA5 Historical Weather API](https://open-meteo.com/)
- **Analysis & Development**: Senior Data Science & Engineering Project
