import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Weather Forecast",
    page_icon="🌤️",
    layout="wide"
)


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "predictor"
    / "final_weather_model.pkl"
)

DATA_PATH = BASE_DIR / "weather_cleaned.csv"

# =========================================================
# Load Model
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# Fourier Features
# Same function used in Task 5
# =========================================================

def create_fourier_features(
    index,
    period=365.25,
    K=5
):

    t = np.arange(len(index))

    features = {}

    for k in range(1, K + 1):

        features[f"sin_{k}"] = np.sin(
            2 * np.pi * k * t / period
        )

        features[f"cos_{k}"] = np.cos(
            2 * np.pi * k * t / period
        )

    return pd.DataFrame(
        features,
        index=index
    )


# =========================================================
# Custom CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f8fafc;
    }

    .title {
        text-align: center;
        color: #0f172a;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .forecast-box {
        background-color: #eff6ff;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #bfdbfe;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Header
# =========================================================

st.markdown(
    '<div class="title">🌤️ Weather Forecast</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Predict the next 30 days of temperature using
    a SARIMAX time-series forecasting model.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Prediction Button
# =========================================================

if st.button(
    "Predict Next 30 Days",
    use_container_width=True
):

    try:

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
        # Forecast Settings
        # -------------------------------------------------

        forecast_days = 30

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

        results = []

        for i in range(forecast_days):

            results.append({
                "Date": future_index[i].strftime("%Y-%m-%d"),

                "Forecast Temperature (°C)": round(
                    float(future_pred.iloc[i]),
                    2
                ),

                "Lower 95% CI (°C)": round(
                    float(future_ci.iloc[i, 0]),
                    2
                ),

                "Upper 95% CI (°C)": round(
                    float(future_ci.iloc[i, 1]),
                    2
                )
            })


        forecast_df = pd.DataFrame(results)


        # =================================================
        # Results
        # =================================================

        st.success(
            "30-day temperature forecast generated successfully!"
        )

        st.subheader("30-Day Temperature Forecast")

        st.caption(
            "Forecast with 95% confidence intervals"
        )

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # Forecast Chart
        # =================================================

        st.subheader("Forecast Trend")

        chart_df = forecast_df.set_index("Date")

        st.line_chart(
            chart_df[
                "Forecast Temperature (°C)"
            ]
        )


        # =================================================
        # Download Results
        # =================================================

        csv = forecast_df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=csv,
            file_name="weather_30_day_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"❌ An error occurred: {str(e)}"
        )