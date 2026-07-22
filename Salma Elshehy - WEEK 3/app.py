from flask import Flask, render_template, request
import pandas as pd
import joblib

# ==========================================
# Load Model and Scaler
# ==========================================

app = Flask(__name__)

model = joblib.load("TaxiFarePredictionModel.pkl")
scaler = joblib.load("TaxiFareScaler.pkl")


# ==========================================
# Home Page
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

        # ==============================
        # Read User Input
        # ==============================

        passenger_count = int(request.form["passenger_count"])
        hour = int(request.form["hour"])
        day = int(request.form["day"])
        month = int(request.form["month"])
        weekday = int(request.form["weekday"])
        year = int(request.form["year"])

        pickup_longitude = float(request.form["pickup_longitude"])
        pickup_latitude = float(request.form["pickup_latitude"])

        dropoff_longitude = float(request.form["dropoff_longitude"])
        dropoff_latitude = float(request.form["dropoff_latitude"])

        distance = float(request.form["distance"])
        bearing = float(request.form["bearing"])

        jfk_dist = float(request.form["jfk_dist"])
        ewr_dist = float(request.form["ewr_dist"])
        lga_dist = float(request.form["lga_dist"])
        sol_dist = float(request.form["sol_dist"])
        nyc_dist = float(request.form["nyc_dist"])

        car_condition = int(request.form["car_condition"])
        weather = int(request.form["weather"])
        traffic = int(request.form["traffic"])


        # ==============================
        # Validation
        # ==============================

        if passenger_count <= 0:
            return render_template(
                "index.html",
                prediction_text="Passenger count must be greater than zero."
            )

        if hour < 0 or hour > 23:
            return render_template(
                "index.html",
                prediction_text="Hour must be between 0 and 23."
            )

        if day < 1 or day > 31:
            return render_template(
                "index.html",
                prediction_text="Day must be between 1 and 31."
            )

        if month < 1 or month > 12:
            return render_template(
                "index.html",
                prediction_text="Month must be between 1 and 12."
            )

        if weekday < 0 or weekday > 6:
            return render_template(
                "index.html",
                prediction_text="Weekday must be between 0 and 6."
            )

        if year < 2000 or year > 2035:
            return render_template(
                "index.html",
                prediction_text="Please enter a valid year."
            )

        if distance < 0:
            return render_template(
                "index.html",
                prediction_text="Distance cannot be negative."
            )


        # ==============================
        # Feature Engineering
        # ==============================

        is_weekend = 1 if weekday in [5, 6] else 0

        is_night = 1 if hour >= 22 or hour <= 5 else 0

        is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0


        # ==============================
        # Create DataFrame
        # ==============================

        data = pd.DataFrame({

            "Car Condition": [car_condition],
            "Weather": [weather],
            "Traffic Condition": [traffic],

            "pickup_longitude": [pickup_longitude],
            "pickup_latitude": [pickup_latitude],

            "dropoff_longitude": [dropoff_longitude],
            "dropoff_latitude": [dropoff_latitude],

            "passenger_count": [passenger_count],

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
            "bearing": [bearing],

            "is_weekend": [is_weekend],
            "is_night": [is_night],
            "is_rush_hour": [is_rush_hour]

        })


        # ==============================
        # Scaling
        # ==============================

        data_scaled = scaler.transform(data)


        # ==============================
        # Prediction
        # ==============================

        prediction = model.predict(data_scaled)

        return render_template(
            "index.html",
            prediction_text=f"Estimated Taxi Fare: ${prediction[0]:.2f}"
        )

    except ValueError:
        return render_template(
            "index.html",
            prediction_text="Please enter valid numeric values."
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


# ==========================================
# Run App
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)