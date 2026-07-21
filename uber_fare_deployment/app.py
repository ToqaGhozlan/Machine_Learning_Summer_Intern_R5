from flask import Flask, render_template, request
from geopy.geocoders import Nominatim
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from transformers import DegreeConverter, ColumnDropper, CyclicalEncoder, PercentileCapper

app = Flask(__name__)

pipeline = joblib.load('model/fare_prediction_pipeline.pkl')

geolocator = Nominatim(user_agent="uber_fare_app")

# Fixed landmark coordinates (degrees) used to compute distance features
JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
NYC_CENTER = (40.7128, -74.0060)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def geocode_address(address):
    location = geolocator.geocode(address, timeout=10)
    if location is None:
        return None
    return location.latitude, location.longitude


@app.route('/', methods=['GET', 'POST'])
def index():
    error = None

    if request.method == 'POST':
        pickup_address = request.form.get('pickup_address', '').strip()
        dropoff_address = request.form.get('dropoff_address', '').strip()
        trip_datetime_str = request.form.get('trip_datetime', '').strip()

        if not pickup_address or not dropoff_address or not trip_datetime_str:
            error = "Please fill in pickup address, dropoff address, and trip date/time."
            return render_template('index.html', error=error)

        try:
            trip_dt = datetime.strptime(trip_datetime_str, '%Y-%m-%dT%H:%M')
        except ValueError:
            error = "Trip date/time is not in a valid format."
            return render_template('index.html', error=error)

        pickup_coords = geocode_address(pickup_address)
        if pickup_coords is None:
            error = f"Could not find location for pickup address: '{pickup_address}'. Try a more specific address."
            return render_template('index.html', error=error)

        dropoff_coords = geocode_address(dropoff_address)
        if dropoff_coords is None:
            error = f"Could not find location for dropoff address: '{dropoff_address}'. Try a more specific address."
            return render_template('index.html', error=error)

        pickup_lat, pickup_lon = pickup_coords
        dropoff_lat, dropoff_lon = dropoff_coords

        try:
            distance_km = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
            bearing_deg = initial_bearing_deg(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
            jfk_dist = haversine_km(dropoff_lat, dropoff_lon, JFK[0], JFK[1])
            lga_dist = haversine_km(dropoff_lat, dropoff_lon, LGA[0], LGA[1])
            nyc_dist = haversine_km(dropoff_lat, dropoff_lon, NYC_CENTER[0], NYC_CENTER[1])

            input_row = pd.DataFrame([{
                'pickup_longitude': np.radians(pickup_lon),
                'pickup_latitude': np.radians(pickup_lat),
                'dropoff_longitude': np.radians(dropoff_lon),
                'dropoff_latitude': np.radians(dropoff_lat),
                'hour': trip_dt.hour,
                'day': trip_dt.day,
                'month': trip_dt.month,
                'weekday': trip_dt.weekday(),
                'year': trip_dt.year,
                'jfk_dist': jfk_dist,
                'lga_dist': lga_dist,
                'nyc_dist': nyc_dist,
                'distance': distance_km,
                'bearing': bearing_deg,
            }])

            pred_fare = pipeline.predict(input_row)[0]

            if pred_fare < 0 or pred_fare > 500:
                error = "The predicted fare fell outside a sensible range. Please check the addresses entered."
                return render_template('index.html', error=error)

            prediction = round(float(pred_fare), 2)

        except Exception as e:
            error = f"Something went wrong while computing the prediction: {e}"
            return render_template('index.html', error=error)

        return render_template(
            'result.html',
            pickup_address=pickup_address,
            dropoff_address=dropoff_address,
            trip_datetime=trip_dt.strftime('%Y-%m-%d %H:%M'),
            distance_km=round(distance_km, 2),
            jfk_dist=round(jfk_dist, 2),
            lga_dist=round(lga_dist, 2),
            nyc_dist=round(nyc_dist, 2),
            bearing_deg=round(bearing_deg, 1),
            prediction=prediction,
        )

    return render_template('index.html', error=error)


if __name__ == '__main__':
    app.run(debug=True)