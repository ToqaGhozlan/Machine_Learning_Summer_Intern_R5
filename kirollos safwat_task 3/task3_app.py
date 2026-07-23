from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

pipeline = joblib.load("final_pipeline.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    passenger_count = int(request.form["passenger_count"])
    weather = request.form["weather"]
    traffic = request.form["traffic"]
    car_condition = int(request.form["car_condition"])
    pickup_latitude = float(request.form["pickup_latitude"])
    pickup_longitude = float(request.form["pickup_longitude"])

    dropoff_latitude = float(request.form["dropoff_latitude"])
    dropoff_longitude = float(request.form["dropoff_longitude"])
    pickup_datetime = request.form["pickup_datetime"]
    distance = float(request.form["distance"])
    dt = datetime.strptime(pickup_datetime, "%Y-%m-%dT%H:%M")
    hour = dt.hour
    month = dt.month
    year = dt.year
    bearing = 17.433258
    jfk_dist = 41.825498
    ewr_dist = 30.306285
    lga_dist = 22.123631
    sol_dist = 18.853499
    nyc_dist = 11.114646

    weather_rainy = 1 if weather == "rainy" else 0
    weather_stormy = 1 if weather == "stormy" else 0
    weather_sunny = 1 if weather == "sunny" else 0
    weather_windy = 1 if weather == "windy" else 0

    traffic_dense = 1 if traffic == "Dense Traffic" else 0
    traffic_flow = 1 if traffic == "Flow Traffic" else 0

    input_data = pd.DataFrame([{
    "Car Condition": car_condition,
    "pickup_longitude": pickup_longitude,
    "pickup_latitude": pickup_latitude,
    "dropoff_longitude": dropoff_longitude,
    "dropoff_latitude": dropoff_latitude,
    "passenger_count": passenger_count,
    "hour": hour,
    "month": month,
    "year": year,
    "jfk_dist": jfk_dist,
    "ewr_dist": ewr_dist,
    "lga_dist": lga_dist,
    "sol_dist": sol_dist,
    "nyc_dist": nyc_dist,
    "distance": distance,
    "bearing": bearing,
    "Weather_rainy": weather_rainy,
    "Weather_stormy": weather_stormy,
    "Weather_sunny": weather_sunny,
    "Weather_windy": weather_windy,
    "Traffic Condition_Dense Traffic": traffic_dense,
    "Traffic Condition_Flow Traffic": traffic_flow
    }])

    prediction = pipeline.predict(input_data)[0]

    return render_template(
    "index.html",
    prediction=round(prediction, 2)
)


if __name__ == "__main__":
    app.run(debug=True)