"""Seasonal climate normals for Alexandria, Egypt, used to auto-fill the
SARIMA model's exogenous weather inputs when the user only supplies a date.

The trained model needs a value for each of WEATHER_FEATURES on every day
it forecasts. Since the app no longer asks the user to type these in, we
estimate them from typical (climatological) conditions for that time of
year instead.

Sources (30-year monthly normals, 1990-2021):
- Max/min temperature, humidity, wind speed, precipitation:
  timeanddate.com climate averages for Alexandria, Egypt.
- Solar radiation (kWh/m2/day): Alexandria-specific monthly values from
  "Simulation and Estimation of Daily Global Solar Radiation in Egypt"
  (station at 31°13'N, 29°58'E).

These are typical values for the time of year, not real forecasts -
actual daily weather can vary a lot from the seasonal average. Treat
predictions from this input mode as an illustration of the model's
seasonal behavior rather than a real weather forecast.
"""
from datetime import date, timedelta

# Monthly normals, values represent the middle of each month (the 15th).
# precipitation is expressed as average mm per day within that month
# (monthly total / days in month), since the model expects a daily value.
MONTHLY_NORMALS = {
    1:  {"max_temperature": 18.3, "min_temperature": 9.4,  "precipitation": 2.17, "humidity": 71, "wind_speed": 6.26, "solar_radiation": 2.16},
    2:  {"max_temperature": 18.9, "min_temperature": 9.4,  "precipitation": 1.23, "humidity": 69, "wind_speed": 6.71, "solar_radiation": 3.11},
    3:  {"max_temperature": 21.1, "min_temperature": 11.7, "precipitation": 0.54, "humidity": 67, "wind_speed": 7.60, "solar_radiation": 4.50},
    4:  {"max_temperature": 23.9, "min_temperature": 13.9, "precipitation": 0.15, "humidity": 65, "wind_speed": 7.60, "solar_radiation": 5.77},
    5:  {"max_temperature": 26.7, "min_temperature": 17.8, "precipitation": 0.03, "humidity": 66, "wind_speed": 7.60, "solar_radiation": 6.75},
    6:  {"max_temperature": 28.9, "min_temperature": 21.7, "precipitation": 0.00, "humidity": 68, "wind_speed": 8.05, "solar_radiation": 7.45},
    7:  {"max_temperature": 30.6, "min_temperature": 23.9, "precipitation": 0.00, "humidity": 69, "wind_speed": 8.49, "solar_radiation": 7.31},
    8:  {"max_temperature": 31.1, "min_temperature": 24.4, "precipitation": 0.03, "humidity": 69, "wind_speed": 8.05, "solar_radiation": 6.68},
    9:  {"max_temperature": 30.0, "min_temperature": 22.8, "precipitation": 0.03, "humidity": 66, "wind_speed": 7.15, "solar_radiation": 5.46},
    10: {"max_temperature": 27.8, "min_temperature": 19.4, "precipitation": 0.34, "humidity": 66, "wind_speed": 6.26, "solar_radiation": 4.00},
    11: {"max_temperature": 23.9, "min_temperature": 15.0, "precipitation": 1.10, "humidity": 69, "wind_speed": 5.81, "solar_radiation": 2.55},
    12: {"max_temperature": 20.6, "min_temperature": 11.1, "precipitation": 1.74, "humidity": 72, "wind_speed": 5.81, "solar_radiation": 1.92},
}

FEATURES = ["max_temperature", "min_temperature", "precipitation", "humidity", "wind_speed", "solar_radiation"]


def _month_anchor(year: int, month: int) -> date:
    """The 15th of the given month - the date each monthly normal represents."""
    return date(year, month, 15)


def estimate_weather_for_date(d: date) -> dict:
    """Estimate typical weather for the given date by linearly interpolating
    between the two nearest months' normals (wrapping around December -> January)."""
    if d.day >= 15:
        m1, y1 = d.month, d.year
        m2, y2 = (d.month + 1, d.year) if d.month < 12 else (1, d.year + 1)
    else:
        m1, y1 = (d.month - 1, d.year) if d.month > 1 else (12, d.year - 1)
        m2, y2 = d.month, d.year

    anchor1 = _month_anchor(y1, m1)
    anchor2 = _month_anchor(y2, m2)
    total_days = (anchor2 - anchor1).days
    frac = (d - anchor1).days / total_days if total_days else 0.0

    n1 = MONTHLY_NORMALS[m1]
    n2 = MONTHLY_NORMALS[m2]

    return {
        feat: round(n1[feat] + (n2[feat] - n1[feat]) * frac, 2)
        for feat in FEATURES
    }
