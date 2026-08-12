from django import forms

class WeatherPredictionForm(forms.Form):
    temp_mean = forms.FloatField(
        label="Temperature (°C)",
        min_value=-10,
        max_value=60,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': 'any',
            'id': 'temp_mean'
        }),
        error_messages={'required': 'Temperature is required.'}
    )
    humidity = forms.FloatField(
        label="Humidity (%)",
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': 'any',
            'id': 'humidity'
        }),
        error_messages={'required': 'Humidity is required.'}
    )
    wind_speed = forms.FloatField(
        label="Wind Speed (km/h)",
        min_value=0,
        max_value=250,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': 'any',
            'id': 'wind_speed'
        }),
        error_messages={'required': 'Wind speed is required.'}
    )
    precipitation = forms.FloatField(
        label="Precipitation (mm)",
        min_value=0,
        max_value=500,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': 'any',
            'id': 'precipitation'
        }),
        error_messages={'required': 'Precipitation is required.'}
    )
