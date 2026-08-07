# Taxi Fare Prediction Web Application

## Project Description

This project predicts taxi fare prices using a trained Machine Learning model (Random Forest Regressor). The application was built using Flask and applies the same preprocessing pipeline used during model training before making predictions.

---

## Project Files

- app.py
- TaxiFarePredictionModel.pkl
- TaxiFareScaler.pkl
- templates/index.html
- static/style.css
---

## Requirements

- Python 3.10 or later
- Flask
- pandas
- numpy
- scikit-learn
- joblib

Install the required packages using:

```bash
pip install flask pandas numpy scikit-learn joblib
```

---

## How to Run the Application

1. Open a terminal in the project folder.

2. Run the Flask application:

```bash
python app.py
```

3. Open your browser and go to:

```
http://127.0.0.1:5000
```

4. Fill in the trip information.

5. Click **Predict Fare** to get the estimated taxi fare.

---

## Machine Learning Pipeline

The application uses the same preprocessing pipeline as the training notebook:

- Input validation
- Feature Engineering
  - is_weekend
  - is_night
  - is_rush_hour
- Feature Scaling using StandardScaler
- Prediction using the trained Random Forest model

---

## Author

Salma Elshehy