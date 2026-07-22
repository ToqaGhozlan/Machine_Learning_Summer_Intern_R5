"""
validate_formulas.py
=====================
Run this ONCE against your own final_internship_data.csv — the file has both
the raw pickup/dropoff coordinates AND the pre-engineered `distance` / `bearing`
/ `nyc_dist` columns together, so we can check which formula/unit actually
reproduces them, instead of guessing.

feature_engineering.py currently assumes:
  - distance / nyc_dist: haversine great-circle distance in KILOMETRES (R=6371)
  - bearing: atan2(y, x) with dlon = pickup_lon − dropoff_lon

This script prints the MAE for that assumption *and* the alternatives (miles,
opposite bearing sign) so you can see in one glance whether the current
assumption is right, or which one to switch to in feature_engineering.py.

Usage:
    python validate_formulas.py path/to/final_internship_data.csv
"""
import sys

import numpy as np
import pandas as pd

NYC_LAT, NYC_LON = 40.7128, -74.0060


def haversine(lat1, lon1, lat2, lon2, radius):
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2, dlon_sign):
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlon = (lon1 - lon2) if dlon_sign == 1 else (lon2 - lon1)
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "final_internship_data.csv"
    print(f"Loading {path} ...")
    df = pd.read_csv(path)

    needed = [
        "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
        "distance", "bearing", "nyc_dist",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Missing expected columns: {missing}. Can't validate — check your CSV's column names.")
        return

    df = df.dropna(subset=needed)
    sample = df.sample(min(20000, len(df)), random_state=42)
    print(f"Validating against {len(sample)} sampled rows.\n")

    print("== distance ==")
    for label, R in [("km  (R=6371.0)", 6371.0), ("mi  (R=3958.8)", 3958.8)]:
        pred = haversine(sample.pickup_latitude, sample.pickup_longitude,
                          sample.dropoff_latitude, sample.dropoff_longitude, R)
        mae = (pred - sample.distance).abs().mean()
        print(f"  {label}: MAE = {mae:.6f}")

    print("\n== nyc_dist (distance to NYC center 40.7128, -74.0060) ==")
    for label, R in [("km  (R=6371.0)", 6371.0), ("mi  (R=3958.8)", 3958.8)]:
        pred = haversine(sample.pickup_latitude, sample.pickup_longitude, NYC_LAT, NYC_LON, R)
        mae = (pred - sample.nyc_dist).abs().mean()
        print(f"  {label}: MAE = {mae:.6f}")

    print("\n== bearing ==")
    for label, sign in [("dlon = pickup_lon − dropoff_lon", 1), ("dlon = dropoff_lon − pickup_lon", -1)]:
        pred = bearing(sample.pickup_latitude, sample.pickup_longitude,
                        sample.dropoff_latitude, sample.dropoff_longitude, sign)
        mae = (pred - sample.bearing).abs().mean()
        print(f"  {label}: MAE = {mae:.6f}")

    print(
        "\nWhichever line has the lowest MAE (ideally ~0) in each section is the "
        "formula your dataset actually uses. If it's not the km / dlon=pickup−dropoff "
        "combination that feature_engineering.py currently assumes, update "
        "EARTH_RADIUS_KM and the bearing() dlon sign there to match."
    )


if __name__ == "__main__":
    main()
