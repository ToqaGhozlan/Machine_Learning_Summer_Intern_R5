# Implementation Plan: Friendly Fields for Uber Fare Prediction

## Steps

- [x] Step 0: Understand the codebase and data categories
- [x] Step 1: Update `requirements.txt` — add `requests` for Nominatim geocoding
- [x] Step 2: Update `uber_fare_pipeline.py`:
  - [x] Fix `validate_trip_payload` allowed categories to match training data
- [x] Step 3: Update `app.py`:
  - [x] Add geocoding function using Nominatim (degrees)
  - [x] Add degrees→radians conversion for coordinates
  - [x] Pre-compute derived features (Distance, Bearing, airport distances, IsWeekend) from degree coords
  - [x] Update `/` route defaults to friendly field names
  - [x] Update `/predict` route for friendly fields + geocoding + model payload
- [x] Step 4: Update `templates/index.html`:
  - [x] Replace coordinate fields with friendly fields (name, date parts, addresses, etc.)
  - [x] Update select options to match training data categories
- [x] Step 5: Update `tests/test_validation.py`:
  - [x] Update test payloads to match corrected categories
- [x] Step 6: Final verification — run tests and verify end-to-end

