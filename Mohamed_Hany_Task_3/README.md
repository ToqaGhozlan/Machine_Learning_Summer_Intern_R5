# 🚕 Uber Fare Prediction – NYC

A production-style Machine Learning project that predicts Uber fares in New York City using a CatBoost regression model deployed with Flask.

The project demonstrates an end-to-end ML workflow including data preprocessing, feature engineering, model comparison, training, deployment, and a user-friendly web interface.

---

# 📌 Project Overview

This application predicts the estimated Uber fare based on:

* 📍 Pickup Location
* 🏁 Dropoff Location
* 🕒 Pickup Date & Time
* 👥 Passenger Count
* 🌤 Weather Condition
* 🚗 Car Condition
* 🚦 Traffic Condition

The trained model is exposed through a Flask web application where users can enter trip information and receive fare predictions in real time.

---

# 🎯 Project Objectives

* Build and compare multiple regression models.
* Select the best-performing model using objective evaluation metrics.
* Deploy the trained model as a Flask web application.
* Ensure identical preprocessing during both training and inference.
* Validate user inputs to prevent invalid predictions.

---

# 🛠 Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost

### Backend

* Flask
* Joblib

### Frontend

* HTML5
* CSS3
* JavaScript
* Leaflet.js
* OpenStreetMap

---

# 📂 Project Structure

```text
Uber-Fare-Prediction/
│
├── app.py
├── train.py
├── feature_engineering.py
├── utils.py
├── requirements.txt
├── README.md
│
├── models/
│   └── catboost_fare_pipeline.joblib
│
├── data/
│   └── final_internship_data.csv
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── notebooks/
│   ├── Uber_model.ipynb
│   └── Uber_EDA.ipynb
│
├── screenshots/
│
└── tests/
```

---

# ⚙️ Machine Learning Pipeline

The project follows the complete machine learning workflow:

1. Data Cleaning
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Model Selection
6. Model Serialization
7. Flask Deployment
8. Real-Time Prediction

The same preprocessing pipeline is used during both training and inference, eliminating Training/Serving Skew.

---

# 🔍 Feature Engineering

The application automatically computes:

* Trip Distance (Haversine Formula)
* Bearing
* Distance to JFK Airport
* Distance to LaGuardia Airport
* Distance to NYC City Center
* Cyclic Time Features

  * Hour
  * Weekday
  * Month
* Rush Hour
* Weekend Flag
* Night Flag
* Encoded Traffic Condition
* Encoded Car Condition

---

# 🤖 Model Comparison

The following regression models were evaluated:

| Model                   |      MAE |     RMSE |       R² |
| ----------------------- | -------: | -------: | -------: |
| Linear Regression       |      ... |      ... |      ... |
| Random Forest Regressor |      ... |      ... |      ... |
| **CatBoost Regressor**  | **1.57** | **3.27** | **0.88** |

CatBoost was selected because it achieved the best overall balance between prediction accuracy, robustness, and inference speed on the test dataset.

---

# 🚀 Running the Project

## 1. Clone Repository

```bash
git clone <repository-url>
cd Uber-Fare-Prediction
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Train the Model

```bash
python train.py
```

Optional

```bash
python train.py --tune
```

```bash
python train.py --gpu
```

---

## 5. Run Flask Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5001
```

---

# 🌐 Web Application Features

* Interactive map using Leaflet
* Pickup & Dropoff point selection
* Real-time fare prediction
* Responsive dark UI
* Input validation
* Friendly error messages
* Automatic feature engineering

---

# ✅ Input Validation

The application validates:

* NYC coordinate boundaries
* Passenger count
* Weather values
* Traffic condition
* Car condition
* Date range (2009–2015)

Both the frontend and backend perform validation to ensure reliable predictions.

---

# 📸 Screenshots

## 📸 Application Screenshots

<p align="center">
  <img src="./screenshots/لقطة%20شاشة%202026-07-19%20195435.png" width="45%">
  <img src="./screenshots/لقطة%20شاشة%202026-07-19%20195537.png" width="45%">
</p>

<p align="center">
  <img src="./screenshots/لقطة%20شاشة%202026-07-19%20195557.png" width="45%">
  <img src="./screenshots/لقطة%20شاشة%202026-07-19%20195614.png" width="45%">
</p>


