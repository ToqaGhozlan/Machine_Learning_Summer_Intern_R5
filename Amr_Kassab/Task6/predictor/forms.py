from django import forms


MONTH_CHOICES = [
    (1, 'January'), (2, 'February'), (3, 'March'),
    (4, 'April'), (5, 'May'), (6, 'June'),
    (7, 'July'), (8, 'August'), (9, 'September'),
    (10, 'October'), (11, 'November'), (12, 'December'),
]

HOUR_CHOICES = [(h, f'{h:02d}:00') for h in range(24)]

MODEL_CHOICES = [
    ('gradient_boosting', 'Gradient Boosting Regressor'),
    ('random_forest', 'Random Forest Regressor'),
]


class WeatherPredictionForm(forms.Form):
    relative_humidity = forms.FloatField(
        label='Relative Humidity',
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'placeholder': 'e.g. 65',
            'step': '0.1',
            'id': 'id_relative_humidity',
        }),
        help_text='Percentage (0–100%)',
    )

    precipitation = forms.FloatField(
        label='Precipitation',
        min_value=0,
        max_value=200,
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'placeholder': 'e.g. 0.0',
            'step': '0.1',
            'id': 'id_precipitation',
        }),
        help_text='Millimeters (mm)',
    )

    wind_speed = forms.FloatField(
        label='Wind Speed',
        min_value=0,
        max_value=200,
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'placeholder': 'e.g. 12.5',
            'step': '0.1',
            'id': 'id_wind_speed',
        }),
        help_text='km/h',
    )

    cloud_cover = forms.FloatField(
        label='Cloud Cover',
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'placeholder': 'e.g. 30',
            'step': '1',
            'id': 'id_cloud_cover',
        }),
        help_text='Percentage (0–100%)',
    )

    surface_pressure = forms.FloatField(
        label='Surface Pressure',
        min_value=850,
        max_value=1100,
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'placeholder': 'e.g. 1013.25',
            'step': '0.1',
            'id': 'id_surface_pressure',
        }),
        help_text='Hectopascals (hPa)',
    )

    hour = forms.ChoiceField(
        label='Hour of Day',
        choices=HOUR_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-select',
            'id': 'id_hour',
        }),
    )

    month = forms.ChoiceField(
        label='Month',
        choices=MONTH_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-select',
            'id': 'id_month',
        }),
    )

    model_choice = forms.ChoiceField(
        label='Prediction Model',
        choices=MODEL_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-select',
            'id': 'id_model_choice',
        }),
        required=False,
    )
