# 🌦️ Damietta Weather Analytics & Time-Series EDA

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.15+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time--Series-4B6584?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

A complete **Time-Series Exploratory Data Analysis (EDA)** and **interactive
analytics dashboard** covering **731 daily weather observations (2024–2025)** for
**Damietta, Egypt**, retrieved from the **NASA POWER API** — built for Cellula
Technologies' ML Internship Program, Task 4.

---

## 📌 Executive Summary

### 🌟 Key Findings
- **Gentle coastal thermal regime:** annual mean temperature is **22.5 °C**, ranging
  from a winter low of **12.4 °C** (23 Feb 2025) to a summer peak of **30.8 °C**
  (11 Sep 2024) — a noticeably gentler swing than an inland desert station, thanks to
  Damietta's Mediterranean coastal position.
- **Winter-concentrated rainfall:** total precipitation across both years is
  **125.6 mm**, with **518 of 731 days (71%) completely dry**. **69%** of all
  rainfall falls in the Nov–Mar window — a wet-winter / dry-summer coastal profile.
- **Weak temperature–humidity coupling:** temperature correlates only weakly with
  humidity (**r = -0.09**) and wind speed (**r = -0.13**) — much weaker than a desert
  station, since sea proximity keeps humidity elevated year-round regardless of season.
- **Non-stationarity confirmed:** Augmented Dickey-Fuller test on raw daily
  temperature gives **statistic = -1.42, p = 0.572 > 0.05** — fail to reject the
  unit-root null, i.e. the series is non-stationary and dominated by its annual
  seasonal cycle.

---

## 📊 Interactive Dashboard Preview

The repository includes a standalone, self-contained **interactive web dashboard**
(`dashboard.html`) with **20 Plotly visualizations** across 6 analytical tabs:

1. **📊 Distributions** — histograms + marginal boxplots and violin/KDE plots for all
   4 core meteorological variables.
2. **📈 Temporal Trends** — daily temperature with 7-/30-day rolling means, daily
   rainfall, daily max/min temperature trends.
3. **🔬 Decomposition & ACF** — additive seasonal decomposition (trend / seasonal /
   residual, period=365) and ACF/PACF (60 lags).
4. **🔗 Correlations** — correlation heatmap, 4-variable scatter matrix, 3D
   interaction plot (Temp × Humidity × Wind), lag plots (k=1, k=7).
5. **🗓️ Seasonal Analysis** — monthly boxplots (temperature & precipitation),
   parallel coordinates, calendar heatmap by day-of-year.
6. **⚠️ Volatility & Outliers** — 7-day rolling std. dev. volatility, rolling
   Z-score outlier detection (|Z| > 2.5), the largest single-day rainfall event
   highlighted.

---

## 🛠️ Technical Workflow

### 1. Data Acquisition (`Part 1`)
- **Source:** NASA POWER `temporal/daily/point` API (`power.larc.nasa.gov`) —
  no key/signup required.
- **Coordinates:** Damietta, Egypt (31.4165°N, 31.8133°E).
- **Time range:** 2024-01-01 → 2025-12-31 (731 daily observations, two full
  calendar years — required so `seasonal_decompose(period=365)` has the ≥730
  observations it needs to run at all).
- **Data quality:** 100% complete — 0 missing values, 0 duplicate timestamps,
  0 `-999` sentinel gaps, daily frequency (`freq='D'`) confirmed.

### 2. Time-Series Structuring & Cleaning (`Part 2`)
- Datetime index, gap/duplicate/frequency checks, missing-value strategy,
  unit-consistency confirmation (°C, mm/day, %, m/s — all SI, no conversion needed).

### 3. Decomposition & Stationarity (`Part 3.3`)
- **Seasonal decomposition** (additive, `period=365`): trend, seasonal, residual.
- **ADF test:**
  ```
  ADF Statistic: -1.4216
  p-value      : 0.5719  (Fail to reject H0 -> Non-Stationary)
  Critical Values: 1% (-3.4394), 5% (-2.8655), 10% (-2.5689)
  ```
- **ACF/PACF (60 lags):** slow-decaying ACF, PACF cutting off after lag 1–2.

### 4. Relationships & Anomalies (`Part 3.4`)
- **Correlation matrix:** temp–humidity r=-0.09, temp–precip r=-0.12,
  temp–wind r=-0.13 — all weak, consistent with coastal moisture stability.
- **Rolling Z-score outlier detection** (30-day local window, |Z| > 2.5) plus an
  IQR-style physical-plausibility check (0 implausible temps, 0 negative rainfall).
- **Notable anomaly:** a **15.5 mm storm on 7 Sep 2025** — the single wettest day
  in the two-year record, and an unusually early-autumn rainfall event.

---

## 📂 Repository Structure

```
Task4_Weather_TimeSeries_EDA/
├── Task4_Weather_TimeSeries_EDA.ipynb   # Full notebook (Parts 1-3 + written summary), pre-executed
├── dashboard.html                       # Self-contained interactive Plotly dashboard
├── data/
│   ├── damietta_2024_2025_raw.json      # Raw NASA POWER API response
│   └── damietta_2024_2025_clean.csv     # Cleaned daily dataset (731 rows x 9 cols)
├── requirements.txt                     # Python dependency manifest
└── README.md                            # This file
```

---

## 🚀 How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the notebook
```bash
jupyter notebook Task4_Weather_TimeSeries_EDA.ipynb
```
Run all cells top to bottom. The first code cell in Part 1 tries the live NASA
POWER API first and automatically falls back to the cached `data/*_raw.json` if the
network call fails (e.g. a firewalled sandbox) — so it reproduces identically either
way.

### 3. View the interactive dashboard
Open `dashboard.html` directly in any browser, or host it locally:
```bash
python -m http.server 8000
```
then visit `http://localhost:8000/dashboard.html`.

---

## 📜 License

Distributed under the **MIT License**.

---

## ✍️ Author & Credits

- **Data Source:** [NASA POWER API](https://power.larc.nasa.gov/)
- **Analysis & Development:** Cellula Technologies ML Internship Program — Task 4
