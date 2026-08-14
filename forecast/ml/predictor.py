# forecast/ml/predictor.py

import numpy as np
from forecast.ml.model_loader import model, scaler, LOOKBACK


def predict_next_day(recent_values):
    """
    Predict the next day's T2M given the most recent daily values.

    Args:
        recent_values: list or array of the last LOOKBACK (365) daily
                        T2M values, in chronological order (oldest to newest).

    Returns:
        float: predicted T2M for the next day.
    """

    recent_values = np.array(recent_values, dtype=float)

    if len(recent_values) != LOOKBACK:
        raise ValueError(
            f"Expected exactly {LOOKBACK} values, got {len(recent_values)}"
        )

    # Scale using the SAME scaler fitted during training
    scaled = scaler.transform(recent_values.reshape(-1, 1))

    # Reshape to LSTM input: [samples, timesteps, features]
    X = scaled.reshape(1, LOOKBACK, 1)

    # Predict (scaled)
    pred_scaled = model.predict(X, verbose=0)

    # Inverse transform back to °C
    pred = scaler.inverse_transform(pred_scaled)

    return float(pred[0][0])