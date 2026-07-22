"""
sanity_check.py
----------------
Run this LOCALLY, next to app.py, uber_fare_pipeline.joblib, and
feature_engineering.py, with final_internship_data.csv also present.

It does exactly what the PDF asks for: "test with values you can
sanity-check by hand". It does NOT call the Flask server — it calls the
same preprocess_and_predict() function app.py uses, so you're testing the
real pipeline end-to-end.

Usage:
    python sanity_check.py
"""

import numpy as np
import pandas as pd
import joblib

from feature_engineering import preprocess_and_predict, InputValidationError

MODEL_PATH = "uber_fare_pipeline.joblib"
CSV_PATH = r"C:\Users\USER\Downloads\final_internship_data.csv"


def row_to_form(row):
    """Convert one raw CSV row into the same dict of strings the Flask
    form would submit. NOTE: the raw CSV stores Car Condition / Weather /
    Traffic Condition as plain text columns (not yet encoded) — the
    encoded versions (car_cond_enc, Traffic_Encoded, weather dummies) only
    exist after the notebook's preprocessing step, not in the raw file."""
    dt = pd.Timestamp(year=int(row["year"]), month=int(row["month"]),
                       day=int(row["day"]), hour=int(row["hour"]))
    return {
        "pickup_datetime": dt.strftime("%Y-%m-%dT%H:%M"),
        "pickup_latitude": str(np.degrees(row["pickup_latitude"])),
        "pickup_longitude": str(np.degrees(row["pickup_longitude"])),
        "dropoff_latitude": str(np.degrees(row["dropoff_latitude"])),
        "dropoff_longitude": str(np.degrees(row["dropoff_longitude"])),
        "passenger_count": str(int(row["passenger_count"])),
        "car_condition": row["Car Condition"],           # already "Bad"/"Good"/"Very Good"/"Excellent"
        "weather": str(row["Weather"]).capitalize(),      # raw is lowercase ("windy" -> "Windy")
        "traffic_condition": row["Traffic Condition"],    # already "Flow Traffic"/"Dense Traffic"/"Congested Traffic"
    }


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(CSV_PATH)

    # ---- 1. real rows vs. their real fare_amount ----------------------
    section("1) Real trips: predicted vs. actual fare_amount")
    sample = df.sample(10, random_state=42)
    errors = []
    for _, row in sample.iterrows():
        form = row_to_form(row)
        try:
            pred, _ = preprocess_and_predict(form, pipeline)
        except InputValidationError as e:
            print("  validation error on a real row (should not happen):", e)
            continue
        actual = row["fare_amount"]
        err = abs(pred - actual)
        errors.append(err)
        print(f"  predicted=${pred:6.2f}  actual=${actual:6.2f}  abs_err=${err:5.2f}")
    print(f"\n  mean abs error over this sample: ${np.mean(errors):.2f}")
    print("  (compare this to the test-set MAE you reported in the notebook —")
    print("   it should be in the same ballpark, not several times larger.)")

    # ---- 2. zero-distance trip -----------------------------------------
    section("2) Edge case: pickup == dropoff (should be near the cheapest fare)")
    zero_trip = {
        "pickup_datetime": "2024-06-15T14:00",
        "pickup_latitude": "40.7128", "pickup_longitude": "-74.0060",
        "dropoff_latitude": "40.7128", "dropoff_longitude": "-74.0060",
        "passenger_count": "1", "car_condition": "Good",
        "weather": "Cloudy", "traffic_condition": "Flow Traffic",
    }
    pred, _ = preprocess_and_predict(zero_trip, pipeline)
    print(f"  predicted fare: ${pred:.2f}  (should be a small base-fare-like number)")

    # ---- 3. known long trip: Manhattan -> JFK --------------------------
    section("3) Known long trip: Midtown Manhattan -> JFK Airport")
    jfk_trip = {
        "pickup_datetime": "2024-06-15T14:00",
        "pickup_latitude": "40.7549", "pickup_longitude": "-73.9840",
        "dropoff_latitude": "40.6413", "dropoff_longitude": "-73.7781",
        "passenger_count": "1", "car_condition": "Good",
        "weather": "Cloudy", "traffic_condition": "Flow Traffic",
    }
    pred, _ = preprocess_and_predict(jfk_trip, pipeline)
    print(f"  predicted fare: ${pred:.2f}  (a real Uber JFK trip is roughly $50-70 —")
    print("   this is just a sanity range, not a precise benchmark)")

    # ---- 4. one-variable-at-a-time direction check ---------------------
    section("4) Direction check: does traffic/weather move the fare the right way?")
    base = dict(jfk_trip)
    variants = [
        ("Flow Traffic", "Congested Traffic", "traffic_condition"),
        ("Sunny", "Stormy", "weather"),
        ("Bad", "Excellent", "car_condition"),
    ]
    for low, high, field in variants:
        t_low = dict(base); t_low[field] = low
        t_high = dict(base); t_high[field] = high
        p_low, _ = preprocess_and_predict(t_low, pipeline)
        p_high, _ = preprocess_and_predict(t_high, pipeline)
        print(f"  {field}: {low}=${p_low:.2f}  vs  {high}=${p_high:.2f}  "
              f"(diff={p_high - p_low:+.2f})")

    # ---- 5. invalid input handling --------------------------------------
    section("5) Invalid input: should raise a clean validation error, not crash")
    bad_trip = dict(jfk_trip)
    bad_trip["pickup_latitude"] = "0"  # far outside NYC
    try:
        preprocess_and_predict(bad_trip, pipeline)
        print("  !! no error raised — this should have been rejected !!")
    except InputValidationError as e:
        print("  correctly rejected:", e)


if __name__ == "__main__":
    main()
