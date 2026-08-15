from datetime import timedelta
from django import forms
from django.utils import timezone
from .predictor import last_known_date, MAX_HORIZON_DAYS

MIN_ALLOWED_DATE = last_known_date + timedelta(days=1)
MAX_ALLOWED_DATE = last_known_date + timedelta(days=MAX_HORIZON_DAYS)


class WeatherForecastForm(forms.Form):
    target_date = forms.DateField(
        label="The date you want the temperature forecast for",
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'date-input',
            'min': MIN_ALLOWED_DATE.date().isoformat(),
            'max': MAX_ALLOWED_DATE.date().isoformat(),
        }),
        error_messages={
            'required': 'You must choose a date.',
            'invalid': 'Date format is not recognized.',
        }
    )

    def clean_target_date(self):
        target_date = self.cleaned_data['target_date']
        target_ts = timezone.datetime.combine(target_date, timezone.datetime.min.time())

        if target_ts < MIN_ALLOWED_DATE:
            raise forms.ValidationError(
                f"The date must be after {last_known_date.date()} (the last day we have data for)."
            )
        if target_ts > MAX_ALLOWED_DATE:
            raise forms.ValidationError(
                f"The date is too far in the future. The maximum allowed is {MAX_ALLOWED_DATE.date()} "
                f"({MAX_HORIZON_DAYS} days) so the forecast stays reliable."
            )
        return target_date