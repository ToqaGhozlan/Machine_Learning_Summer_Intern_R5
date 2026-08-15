import datetime as dt

from django import forms

from .ml_model import MAX_FORECAST_HORIZON_DAYS, get_model


class ForecastDateForm(forms.Form):
    target_date = forms.DateField(
        label="Date to forecast",
        widget=forms.DateInput(attrs={
            "type": "date",
            "class": "form-control form-control-lg",
        }),
        error_messages={
            "required": "Please choose a date.",
            "invalid": "That doesn't look like a valid date.",
        },
    )

    def clean_target_date(self):
        target_date = self.cleaned_data["target_date"]
        model = get_model()
        min_date = model.min_valid_date()
        max_date = model.max_valid_date()

        if target_date < min_date:
            raise forms.ValidationError(
                f"Choose a date on or after {min_date.isoformat()} — "
                f"the model's history ends {model.last_train_date.isoformat()}."
            )
        if target_date > max_date:
            raise forms.ValidationError(
                f"Choose a date on or before {max_date.isoformat()} — "
                f"forecasts beyond {MAX_FORECAST_HORIZON_DAYS} days out are too "
                f"uncertain for this model to report responsibly."
            )
        return target_date
