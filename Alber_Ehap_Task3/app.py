from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2, degrees

app = Flask(__name__)

# ==========================================
# Load Pipeline
# ==========================================

pipeline = joblib.load("Taxi_Fare_Pipeline.pkl")

# ==========================================
# Important Locations
# ==========================================

JFK = (40.6413, -73.7781)
EWR = (40.6895, -74.1745)
LGA = (40.7769, -73.8740)
SOL = (40.6892, -74.0445)
NYC = (40.7128, -74.0060)

# ==========================================
# Distance Function
# ==========================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + \
        cos(lat1) * cos(lat2) * \
        sin(dlon / 2) ** 2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


# ==========================================
# Bearing Function
# ==========================================

def calculate_bearing(lat1, lon1, lat2, lon2):

    lat1 = radians(lat1)
    lat2 = radians(lat2)

    diff = radians(lon2 - lon1)

    x = sin(diff) * cos(lat2)

    y = cos(lat1) * sin(lat2) - \
        sin(lat1) * cos(lat2) * cos(diff)

    return degrees(atan2(x, y))


# ==========================================
# Home
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -----------------------------
        # Read Form
        # -----------------------------

        weather = request.form["weather"].strip()

        traffic = request.form["traffic"].strip()

        condition = request.form["condition"].strip()

        pickup_lat = float(request.form["pickup_latitude"])

        pickup_lon = float(request.form["pickup_longitude"])

        dropoff_lat = float(request.form["dropoff_latitude"])

        dropoff_lon = float(request.form["dropoff_longitude"])

        passenger = int(request.form["passenger_count"])

        pickup_datetime = request.form["pickup_datetime"]

        # -----------------------------
        # Validation
        # -----------------------------

        if passenger < 1 or passenger > 6:
            raise ValueError(
                "Passenger count must be between 1 and 6."
            )

        if not (-90 <= pickup_lat <= 90):
            raise ValueError(
                "Invalid Pickup Latitude."
            )

        if not (-90 <= dropoff_lat <= 90):
            raise ValueError(
                "Invalid Dropoff Latitude."
            )

        if not (-180 <= pickup_lon <= 180):
            raise ValueError(
                "Invalid Pickup Longitude."
            )

        if not (-180 <= dropoff_lon <= 180):
            raise ValueError(
                "Invalid Dropoff Longitude."
            )

        if pickup_datetime == "":
            raise ValueError(
                "Pickup Date & Time is required."
            )

        # -----------------------------
        # Date Features
        # -----------------------------

        dt = datetime.strptime(
            pickup_datetime,
            "%Y-%m-%dT%H:%M"
        )

        hour = dt.hour
        day = dt.day
        month = dt.month
        weekday = dt.weekday()
        year = dt.year

        # -----------------------------
        # Engineered Features
        # -----------------------------

        distance = haversine(
            pickup_lat,
            pickup_lon,
            dropoff_lat,
            dropoff_lon
        )

        bearing = calculate_bearing(
            pickup_lat,
            pickup_lon,
            dropoff_lat,
            dropoff_lon
        )

        jfk_dist = haversine(
            pickup_lat,
            pickup_lon,
            JFK[0],
            JFK[1]
        )

        ewr_dist = haversine(
            pickup_lat,
            pickup_lon,
            EWR[0],
            EWR[1]
        )

        lga_dist = haversine(
            pickup_lat,
            pickup_lon,
            LGA[0],
            LGA[1]
        )

        sol_dist = haversine(
            pickup_lat,
            pickup_lon,
            SOL[0],
            SOL[1]
        )

        nyc_dist = haversine(
            pickup_lat,
            pickup_lon,
            NYC[0],
            NYC[1]
        )

        # -----------------------------
        # DataFrame
        # -----------------------------

        input_df = pd.DataFrame({

            "Car Condition": [condition],

            "Weather": [weather],

            "Traffic Condition": [traffic],

            "pickup_longitude": [pickup_lon],

            "pickup_latitude": [pickup_lat],

            "dropoff_longitude": [dropoff_lon],

            "dropoff_latitude": [dropoff_lat],

            "passenger_count": [passenger],

            "hour": [hour],

            "day": [day],

            "month": [month],

            "weekday": [weekday],

            "year": [year],

            "jfk_dist": [jfk_dist],

            "ewr_dist": [ewr_dist],

            "lga_dist": [lga_dist],

            "sol_dist": [sol_dist],

            "nyc_dist": [nyc_dist],

            "distance": [distance],

            "bearing": [bearing]

        })

        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = pipeline.predict(input_df)[0]

        return render_template(
            "result.html",
            fare=round(prediction, 2)
        )

    except Exception as e:

        return render_template(
            "result.html",
            error=f"Prediction Failed: {str(e)}"
        )


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)