from django import forms


class WeatherPredictionForm(forms.Form):

    temperature = forms.FloatField(

        label="Current temperature (°C)",

        min_value=-50,

        max_value=60,

        widget=forms.NumberInput(
            attrs={
                "step": "0.1",
                "placeholder": "e.g. 28.5"
            }
        ),
    )


    humidity = forms.FloatField(

        label="Humidity (%)",

        min_value=0,

        max_value=100,

        widget=forms.NumberInput(
            attrs={
                "step": "0.1",
                "placeholder": "e.g. 55"
            }
        ),
    )


    wind_speed = forms.FloatField(

        label="Wind speed (km/h)",

        min_value=0,

        max_value=250,

        widget=forms.NumberInput(
            attrs={
                "step": "0.1",
                "placeholder": "e.g. 18"
            }
        ),
    )


    pressure = forms.FloatField(

        label="Pressure (hPa)",

        min_value=850,

        max_value=1100,

        widget=forms.NumberInput(
            attrs={
                "step": "0.1",
                "placeholder": "e.g. 1012"
            }
        ),
    )


    def clean(self):

        cleaned = super().clean()

        temperature = cleaned.get("temperature")

        humidity = cleaned.get("humidity")

        wind_speed = cleaned.get("wind_speed")

        pressure = cleaned.get("pressure")


        if (
            temperature is not None
            and humidity is not None
            and wind_speed is not None
            and pressure is not None
        ):

            if humidity > 95 and temperature > 45:

                raise forms.ValidationError(
                    "The combination of very high humidity "
                    "and extreme temperature is outside "
                    "the supported range."
                )


        return cleaned