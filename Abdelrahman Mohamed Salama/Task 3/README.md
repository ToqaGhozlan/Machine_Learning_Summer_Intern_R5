# Uber Fare Prediction - Flask App

This project is a Flask web application that predicts the fare of an Uber trip using a machine learning model trained in Task 2.

---

## Project Structure

```
app.py                     # Main Flask application
sidebar.py                 # Validates user inputs
map_view.py                # Creates geographic and time-based features
results.py                 # Prepares model input and returns predictions

templates/
    index.html             # Web page

static/
    css/style.css          # Page styling
    js/app.js              # Frontend logic and map interaction

fare_pipeline.pkl          # Trained machine learning pipeline
requirements.txt
README.md
```

---

## Installation

1. Copy your trained model (`fare_pipeline.pkl`) into the project folder.

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Open your browser and go to:

```
http://127.0.0.1:9000
```

---

## How It Works

1. Select the pickup and drop-off locations on the map.
2. Enter the number of passengers, date, and time.
3. Click **Predict Fare**.
4. The application validates the input.
5. Geographic and time-based features are generated.
6. These features are sent to the trained machine learning pipeline.
7. The predicted fare is returned and displayed on the page.

---

## Feature Engineering

Before making a prediction, the application generates several features, including:

- Trip distance (Haversine formula)
- Travel direction (Bearing)
- Distance to:
  - JFK Airport
  - Newark Airport
  - LaGuardia Airport
  - Statue of Liberty
  - NYC Center
- Hour, day, month, and year
- Rush hour indicator
- Weekend indicator

---

## Notes

- The model expects latitude and longitude values in **radians**, so the application converts the map coordinates before prediction.
- Default values are used for **Weather**, **Traffic Condition**, and **Car Condition** because they are not provided by the user.
- The saved pipeline already includes preprocessing steps such as encoding, scaling, and feature selection.

---

## Testing

Make sure to test the application with:

- At least **3 valid trips** and verify the predicted fare.
- At least **1 invalid input**, such as:
  - Missing date
  - Passenger count outside the allowed range
  - Pickup or drop-off outside NYC
  - Same pickup and drop-off location

The application should display a clear error message for invalid inputs.