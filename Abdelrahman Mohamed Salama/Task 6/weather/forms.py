from django import forms


class WeatherInputForm(forms.Form):
    max_temperature = forms.FloatField(
        label="Max Temperature (°C)",
        min_value=-10,
        max_value=50,
        widget=forms.NumberInput(attrs={
            "class": "form-control", "step": "0.1", "placeholder": "e.g. 28.5",
        }),
    )
    min_temperature = forms.FloatField(
        label="Min Temperature (°C)",
        min_value=-10,
        max_value=40,
        widget=forms.NumberInput(attrs={
            "class": "form-control", "step": "0.1", "placeholder": "e.g. 18.0",
        }),
    )
    precipitation = forms.FloatField(
        label="Precipitation (mm)",
        min_value=0,
        max_value=200,
        widget=forms.NumberInput(attrs={
            "class": "form-control", "step": "0.1", "placeholder": "e.g. 0.0",
        }),
    )
    humidity = forms.FloatField(
        label="Humidity (%)",
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={
            "class": "form-control", "step": "0.1", "placeholder": "e.g. 65.0",
        }),
    )
    wind_speed = forms.FloatField(
        label="Wind Speed (m/s)",
        min_value=0,
        max_value=40,
        widget=forms.NumberInput(attrs={
            "class": "form-control", "step": "0.1", "placeholder": "e.g. 3.2",
        }),
    )
    solar_radiation = forms.FloatField(
        label="Solar Radiation (kWh/m²)",
        min_value=0,
        max_value=15,
        widget=forms.NumberInput(attrs={
            "class": "form-control", "step": "0.01", "placeholder": "e.g. 5.4",
        }),
    )

    def clean(self):
        cleaned_data = super().clean()
        min_t = cleaned_data.get("min_temperature")
        max_t = cleaned_data.get("max_temperature")
        if min_t is not None and max_t is not None and min_t > max_t:
            self.add_error("min_temperature", "Min temperature can't exceed max temperature.")
        return cleaned_data
