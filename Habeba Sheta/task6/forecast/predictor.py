import os
import joblib
import numpy as np
import pandas as pd
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, 'sarimax_final.pkl')
DATE_PATH = os.path.join(settings.BASE_DIR, 'last_known_date.pkl')

model = joblib.load(MODEL_PATH)
last_known_date = joblib.load(DATE_PATH)

MAX_HORIZON_DAYS = 180  # حد أقصى للتنبؤ (بعده الموديل مش موثوق)


def predict_temperature(target_date: pd.Timestamp):
    """بتاخد تاريخ مستقبلي وترجع (predicted, ci_low, ci_high, steps_ahead)"""
    steps_ahead = (target_date - last_known_date).days

    future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=steps_ahead, freq='D')
    future_doy = future_dates.dayofyear
    future_exog = pd.DataFrame({
        'doy_sin': np.sin(2 * np.pi * future_doy / 365),
        'doy_cos': np.cos(2 * np.pi * future_doy / 365)
    }, index=future_dates)

    forecast = model.get_forecast(steps=steps_ahead, exog=future_exog)
    predicted = forecast.predicted_mean.iloc[-1]
    ci = forecast.conf_int(alpha=0.05).iloc[-1]

    return round(predicted, 1), round(ci.iloc[0], 1), round(ci.iloc[1], 1), steps_ahead