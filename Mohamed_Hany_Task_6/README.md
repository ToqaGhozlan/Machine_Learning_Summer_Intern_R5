# WeatherCast AI ⚡

**24-Hour Predictive Weather Intelligence Platform for Cairo, Egypt**  
*Built with XGBoost Regressor, Django REST API, and Vanilla JavaScript/CSS SaaS Dashboard.*

---

## 🌟 Overview

WeatherCast AI is a production-grade machine learning platform engineered to forecast ambient temperature (`temperature_2m`) 24 hours into the future ($t + 24\text{h}$) using 168 hours of historical context ($t - 168\text{h} \dots t - 1\text{h}$) and real-time atmospheric measurements from Open-Meteo.

---

## 🏗️ Architecture

```
WeatherCast-AI/
├── backend/                  # Django REST API & Domain Layer
│   ├── config/settings/      # Modular settings (base, dev, prod)
│   ├── weather/              # Weather forecasting app
│   │   ├── api/              # Views, serializers, routes
│   │   ├── domain/           # Data contracts, schemas, exceptions
│   │   ├── services/         # Model, feature, weather services
│   │   └── tests/            # Automated test suite
├── frontend/                 # Dark SaaS Dashboard UI
│   ├── assets/css/           # Custom token-based design system
│   └── assets/js/            # Client modules (api, charts, validation)
├── ml/                       # Machine Learning Lifecycle
│   ├── data/                 # Raw dataset (weather.csv)
│   ├── models/               # Serialized model JSON & metadata
│   └── training/             # Reproducible training pipeline
├── scripts/                  # Verification & smoke test scripts
├── MODEL_CONTRACT.md         # Formal mathematical inference contract
├── ARCHITECTURE.md           # System design & layer specifications
├── API_DOCUMENTATION.md      # REST API documentation
└── PRODUCTION_CHECKLIST.md   # Deployment validation checklist
```

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Verify System & Model Parity

```bash
# Verify feature engineering parity
python scripts/verify_features.py

# Verify model serialization
python scripts/verify_model.py

# Run comprehensive smoke test
python scripts/smoke_test.py
```

### 3. Run Test Suite

```bash
python backend/manage.py test weather -v 2
```

### 4. Start Development Server

```bash
python backend/manage.py runserver 0.0.0.0:8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📐 15-Feature Contract

The model consumes exactly 15 causal features:
1. `apparent_temperature` (Exogenous at $t$)
2. `pressure_msl` (Exogenous at $t$)
3. `relative_humidity_2m` (Exogenous at $t$)
4. `hour_cos` (Cyclical hour)
5. `month_cos` (Cyclical month)
6. `dayofyear_sin` (Cyclical DOY sine)
7. `dayofyear_cos` (Cyclical DOY cosine)
8. `temperature_2m_lag_1` ($t - 1\text{h}$)
9. `temperature_2m_lag_24` ($t - 24\text{h}$)
10. `temperature_2m_lag_72` ($t - 72\text{h}$)
11. `temperature_2m_lag_168` ($t - 168\text{h}$)
12. `temperature_2m_rolling_max_6` ($\max_{6\text{h}}$)
13. `temperature_2m_rolling_max_24` ($\max_{24\text{h}}$)
14. `temperature_2m_rolling_mean_24` ($\text{mean}_{24\text{h}}$)
15. `temperature_2m_rolling_std_24` ($\text{std}_{24\text{h}}, \text{ddof}=1$)
