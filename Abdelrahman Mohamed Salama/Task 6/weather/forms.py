from django import forms

from .ml_model import get_forecast_window


class ForecastDateForm(forms.Form):
    forecast_date = forms.DateField(
        label="Forecast Date",
        widget=forms.DateInput(attrs={
            "class": "form-control", "type": "date",
        }),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bounds depend on the trained model (how many days of data it saw),
        # so they're computed at request time and used both for the HTML
        # date-picker limits and for server-side validation below.
        self.min_date, self.max_date = get_forecast_window()
        self.fields["forecast_date"].widget.attrs.update({
            "min": self.min_date.isoformat(),
            "max": self.max_date.isoformat(),
        })
        self.fields["forecast_date"].help_text = (
            f"Pick a date between {self.min_date:%b %d, %Y} and {self.max_date:%b %d, %Y}."
        )

    def clean_forecast_date(self):
        d = self.cleaned_data["forecast_date"]
        if d < self.min_date or d > self.max_date:
            raise forms.ValidationError(
                f"Date must be between {self.min_date:%b %d, %Y} and {self.max_date:%b %d, %Y}."
            )
        return d
