import datetime as dt

from django import forms

from .ml_models import MODEL_CHOICES

TRAIN_END_DATE = dt.date(2025, 7, 31)


class WeatherPredictionForm(forms.Form):
    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        initial="fourier_auto",
        label="Model",
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    # --- Used by arma_manual (converted to a horizon internally) and by
    #     fourier_auto (used directly to compute seasonal terms) ---
    forecast_date = forms.DateField(
        label="Forecast date",
        required=False,
        initial=TRAIN_END_DATE + dt.timedelta(days=1),
        widget=forms.DateInput(attrs={"class": "form-control", "type": "date"}),
    )
    humidity = forms.FloatField(
        label="Humidity (%)", min_value=0, max_value=100, required=False,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
    )
    tempmax = forms.FloatField(
        label="Max temp (°C)", min_value=-50, max_value=60, required=False,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
    )
    dew = forms.FloatField(
        label="Dew point (°C)", min_value=-50, max_value=40, required=False,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
    )
    tempmin = forms.FloatField(
        label="Min temp (°C)", min_value=-50, max_value=60, required=False,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
    )
    precip = forms.FloatField(
        label="Precipitation (mm)", min_value=0, max_value=500, required=False,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
    )
    cloudcover = forms.FloatField(
        label="Cloud cover (%)", min_value=0, max_value=100, required=False,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
    )
    precipprob = forms.FloatField(
        label="Precipitation probability (%)", min_value=0, max_value=100, required=False,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
    )

    EXOG_FIELDS = ["humidity", "tempmax", "dew", "tempmin", "precip", "cloudcover", "precipprob"]

    def clean(self):
        cleaned_data = super().clean()
        model_choice = cleaned_data.get("model_choice")

        if model_choice == "arma_manual":
            forecast_date = cleaned_data.get("forecast_date")
            if forecast_date is None:
                self.add_error("forecast_date", "Pick the date you want to forecast.")
            elif forecast_date <= TRAIN_END_DATE:
                self.add_error(
                    "forecast_date",
                    f"Pick a date after {TRAIN_END_DATE.isoformat()} (end of training data).",
                )
            return cleaned_data

        # armax_auto / fourier_auto: all 7 covariates are required
        for field in self.EXOG_FIELDS:
            if cleaned_data.get(field) is None:
                self.add_error(field, "This field is required for this model.")

        if model_choice == "fourier_auto":
            if cleaned_data.get("forecast_date") is None:
                self.add_error("forecast_date", "Pick the date you want to forecast.")
            elif cleaned_data["forecast_date"] <= TRAIN_END_DATE:
                self.add_error(
                    "forecast_date",
                    f"Pick a date after {TRAIN_END_DATE.isoformat()} (end of training data).",
                )

        # tempmin shouldn't exceed tempmax - a sanity cross-check, not just a range check
        tempmin = cleaned_data.get("tempmin")
        tempmax = cleaned_data.get("tempmax")
        if tempmin is not None and tempmax is not None and tempmin > tempmax:
            self.add_error("tempmin", "Min temp can't be greater than max temp.")

        return cleaned_data
