from django.shortcuts import render
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "predictor" / "final_weather_model.pkl"
DATA_PATH = BASE_DIR / "weather_cleaned.csv"


# =========================================================
# Load Model Once
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# Fourier Features
# Same function used in Task 5
# =========================================================

def create_fourier_features(index, period=365.25, K=5):

    t = np.arange(len(index))

    features = {}

    for k in range(1, K + 1):

        features[f"sin_{k}"] = np.sin(
            2 * np.pi * k * t / period
        )

        features[f"cos_{k}"] = np.cos(
            2 * np.pi * k * t / period
        )

    return pd.DataFrame(features, index=index)


# =========================================================
# Home Page
# =========================================================

def home(request):

    context = {
        "forecast": None,
        "error": None,
        "forecast_days": 30
    }

    if request.method == "POST":

        try:

            # -------------------------------------------------
            # Get Forecast Horizon
            # -------------------------------------------------

            days_input = request.POST.get("forecast_days", "").strip()

            if not days_input:

                raise ValueError(
                    "Please enter the number of forecast days."
                )

            try:

                forecast_days = int(days_input)

            except ValueError:

                raise ValueError(
                    "Forecast days must be a whole number."
                )


            # -------------------------------------------------
            # Validate Range
            # -------------------------------------------------

            if forecast_days < 1:

                raise ValueError(
                    "Forecast days must be at least 1."
                )

            if forecast_days > 90:

                raise ValueError(
                    "Forecast horizon cannot exceed 90 days."
                )


            context["forecast_days"] = forecast_days


            # -------------------------------------------------
            # Load Weather Data
            # -------------------------------------------------

            df = pd.read_csv(
                DATA_PATH,
                parse_dates=["date"],
                index_col="date"
            )

            df = df.asfreq("D")


            # -------------------------------------------------
            # Create Future Dates
            # -------------------------------------------------

            future_index = pd.date_range(
                start=df.index.max() + pd.Timedelta(days=1),
                periods=forecast_days,
                freq="D"
            )


            # -------------------------------------------------
            # Create Fourier Features
            # -------------------------------------------------

            fourier_future = create_fourier_features(
                future_index,
                period=365.25,
                K=5
            )


            # -------------------------------------------------
            # Generate Forecast
            # -------------------------------------------------

            future_forecast = model.get_forecast(
                steps=forecast_days,
                exog=fourier_future
            )


            future_pred = future_forecast.predicted_mean

            future_ci = future_forecast.conf_int()


            # -------------------------------------------------
            # Prepare Results
            # -------------------------------------------------

            forecast_results = []

            for i in range(forecast_days):

                forecast_results.append({

                    "date": future_index[i].strftime(
                        "%Y-%m-%d"
                    ),

                    "temperature": round(
                        float(future_pred.iloc[i]),
                        2
                    ),

                    "lower": round(
                        float(future_ci.iloc[i, 0]),
                        2
                    ),

                    "upper": round(
                        float(future_ci.iloc[i, 1]),
                        2
                    )

                })


            context["forecast"] = forecast_results


        except Exception as e:

            context["error"] = str(e)


    return render(
        request,
        "predictor/home.html",
        context
    )