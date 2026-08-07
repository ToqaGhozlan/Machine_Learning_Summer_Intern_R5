"""
validate_formulas.py
====================
Step 2: Validate that our recalculated features match the original dataset columns.
Compares: distance, bearing, jfk_dist, lga_dist, nyc_dist
Reports: MAE, RMSE, Pearson Correlation for each formula variant.
Goal: find the EXACT formula used during training (zero skew).
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

CSV = "final_internship_data.csv"
EARTH_R = 6371.0

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df = df.dropna(subset=["pickup_longitude","pickup_latitude",
                        "dropoff_longitude","dropoff_latitude",
                        "distance","bearing","jfk_dist","lga_dist","nyc_dist"])

lat1 = df["pickup_latitude"].values
lon1 = df["pickup_longitude"].values
lat2 = df["dropoff_latitude"].values
lon2 = df["dropoff_longitude"].values

def metrics(orig, calc, name):
    mae  = np.mean(np.abs(orig - calc))
    rmse = np.sqrt(np.mean((orig - calc)**2))
    corr, _ = pearsonr(orig, calc)
    print(f"  {name:50s}  MAE={mae:.6f}  RMSE={rmse:.6f}  corr={corr:.8f}")
    return mae, rmse, corr

# ════════════════════════════════════════════════════════════════════════════
# 1. DISTANCE
# ════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("DISTANCE  (original column: 'distance', km)")
print("=" * 80)
orig_dist = df["distance"].values

# Formula A: standard Haversine pickup→dropoff
dlat_A = lat2 - lat1
dlon_A = lon2 - lon1
a_A = np.sin(dlat_A/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon_A/2)**2
dist_A = EARTH_R * 2 * np.arctan2(np.sqrt(a_A), np.sqrt(1-a_A))
r_A = metrics(orig_dist, dist_A, "A: Haversine (pickup→dropoff)")

# Formula B: Haversine with abs(dlon)
dlon_B = np.abs(lon2 - lon1)
dlat_B = np.abs(lat2 - lat1)
a_B = np.sin(dlat_B/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon_B/2)**2
dist_B = EARTH_R * 2 * np.arctan2(np.sqrt(a_B), np.sqrt(1-a_B))
r_B = metrics(orig_dist, dist_B, "B: Haversine abs(dlon)")

# Formula C: Euclidean in radian space * R
dist_C = EARTH_R * np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2)
r_C = metrics(orig_dist, dist_C, "C: Euclidean radian*R")

# Formula D: Manhattan
dist_D = EARTH_R * (np.abs(lat2-lat1) + np.abs(lon2-lon1))
r_D = metrics(orig_dist, dist_D, "D: Manhattan radian*R")

# Formula E: Haversine dropoff→pickup (reversed)
dlat_E = lat1 - lat2
dlon_E = lon1 - lon2
a_E = np.sin(dlat_E/2)**2 + np.cos(lat2)*np.cos(lat1)*np.sin(dlon_E/2)**2
dist_E = EARTH_R * 2 * np.arctan2(np.sqrt(a_E), np.sqrt(1-a_E))
r_E = metrics(orig_dist, dist_E, "E: Haversine (dropoff→pickup)")

print()
best_dist = min([r_A,r_B,r_C,r_D,r_E], key=lambda x: x[0])
print(f"Best MAE for distance: {best_dist[0]:.8f}")

# ════════════════════════════════════════════════════════════════════════════
# 2. BEARING
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("BEARING  (original column: 'bearing', radians)")
print("=" * 80)
orig_bear = df["bearing"].values

# Formula A: standard forward bearing pickup→dropoff
dlon_fwd = lon2 - lon1
y_A = np.sin(dlon_fwd) * np.cos(lat2)
x_A = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon_fwd)
bear_A = np.arctan2(y_A, x_A)
metrics(orig_bear, bear_A, "A: standard forward (lon2-lon1)")

# Formula B: reverse bearing (dlon = lon1 - lon2)
dlon_rev = lon1 - lon2
y_B = np.sin(dlon_rev) * np.cos(lat2)
x_B = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon_rev)
bear_B = np.arctan2(y_B, x_B)
metrics(orig_bear, bear_B, "B: reverse dlon (lon1-lon2)")

# Formula C: swap lat role
y_C = np.sin(dlon_fwd) * np.cos(lat1)
x_C = np.cos(lat2)*np.sin(lat1) - np.sin(lat2)*np.cos(lat1)*np.cos(dlon_fwd)
bear_C = np.arctan2(y_C, x_C)
metrics(orig_bear, bear_C, "C: swap lat role fwd")

# Formula D: swap lat role + reverse dlon
y_D = np.sin(dlon_rev) * np.cos(lat1)
x_D = np.cos(lat2)*np.sin(lat1) - np.sin(lat2)*np.cos(lat1)*np.cos(dlon_rev)
bear_D = np.arctan2(y_D, x_D)
metrics(orig_bear, bear_D, "D: swap lat role + reverse dlon")

# ════════════════════════════════════════════════════════════════════════════
# 3. AIRPORT DISTANCES  – find reference coordinates
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("AIRPORT / CITY DISTANCES  (from pickup point)")
print("=" * 80)

def haversine_to_ref(lat1, lon1, ref_lat, ref_lon):
    dlat = ref_lat - lat1
    dlon = ref_lon - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(ref_lat)*np.sin(dlon/2)**2
    return EARTH_R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Known NYC airport coords in degrees → radians
landmarks = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "SOL": (40.5795, -74.1502),  # Staten Island
    "NYC_center": (40.7128, -74.0060),
    "NYC_midtown": (40.7549, -73.9840),
    "NYC_brooklyn": (40.6782, -73.9442),
}

target_map = {
    "jfk_dist": df["jfk_dist"].values,
    "lga_dist": df["lga_dist"].values,
    "nyc_dist": df["nyc_dist"].values,
}

print("\n--- JFK ---")
for name, (lat_deg, lon_deg) in landmarks.items():
    ref_lat = np.deg2rad(lat_deg)
    ref_lon = np.deg2rad(lon_deg)
    calc = haversine_to_ref(lat1, lon1, ref_lat, ref_lon)
    mae = np.mean(np.abs(df["jfk_dist"].values - calc))
    print(f"  vs {name:15s}  MAE={mae:.4f}")

print("\n--- LGA ---")
for name, (lat_deg, lon_deg) in landmarks.items():
    ref_lat = np.deg2rad(lat_deg)
    ref_lon = np.deg2rad(lon_deg)
    calc = haversine_to_ref(lat1, lon1, ref_lat, ref_lon)
    mae = np.mean(np.abs(df["lga_dist"].values - calc))
    print(f"  vs {name:15s}  MAE={mae:.4f}")

print("\n--- NYC ---")
for name, (lat_deg, lon_deg) in landmarks.items():
    ref_lat = np.deg2rad(lat_deg)
    ref_lon = np.deg2rad(lon_deg)
    calc = haversine_to_ref(lat1, lon1, ref_lat, ref_lon)
    mae = np.mean(np.abs(df["nyc_dist"].values - calc))
    print(f"  vs {name:15s}  MAE={mae:.4f}")

# ── Fine-tune the best reference coords by grid search ──────────────────────
print()
print("=" * 80)
print("GRID SEARCH: Find exact reference coords that minimize MAE")
print("=" * 80)

def best_ref(orig_vals, lat_center_deg, lon_center_deg, r=0.3, steps=40):
    best_mae = 1e9
    best_lat = best_lon = 0
    lats = np.linspace(lat_center_deg - r, lat_center_deg + r, steps)
    lons = np.linspace(lon_center_deg - r, lon_center_deg + r, steps)
    for la in lats:
        for lo in lons:
            calc = haversine_to_ref(lat1, lon1, np.deg2rad(la), np.deg2rad(lo))
            mae = np.mean(np.abs(orig_vals - calc))
            if mae < best_mae:
                best_mae = mae
                best_lat, best_lon = la, lo
    return best_lat, best_lon, best_mae

jfk_lat, jfk_lon, jfk_mae = best_ref(df["jfk_dist"].values, 40.64, -73.78)
print(f"JFK best ref: lat={jfk_lat:.4f} lon={jfk_lon:.4f}  MAE={jfk_mae:.6f}")

lga_lat, lga_lon, lga_mae = best_ref(df["lga_dist"].values, 40.78, -73.87)
print(f"LGA best ref: lat={lga_lat:.4f} lon={lga_lon:.4f}  MAE={lga_mae:.6f}")

nyc_lat, nyc_lon, nyc_mae = best_ref(df["nyc_dist"].values, 40.71, -74.01)
print(f"NYC best ref: lat={nyc_lat:.4f} lon={nyc_lon:.4f}  MAE={nyc_mae:.6f}")

print()
print("=" * 80)
print("SUMMARY - USE THESE IN feature_engineering.py")
print("=" * 80)
print(f"JFK: ({jfk_lat:.6f}, {jfk_lon:.6f})  -> radians ({np.deg2rad(jfk_lat):.8f}, {np.deg2rad(jfk_lon):.8f})")
print(f"LGA: ({lga_lat:.6f}, {lga_lon:.6f})  -> radians ({np.deg2rad(lga_lat):.8f}, {np.deg2rad(lga_lon):.8f})")
print(f"NYC: ({nyc_lat:.6f}, {nyc_lon:.6f})  -> radians ({np.deg2rad(nyc_lat):.8f}, {np.deg2rad(nyc_lon):.8f})")
