from django import forms

from . import ml


class ForecastForm(forms.Form):
    model_choice = forms.ChoiceField(
        choices=ml.MODEL_CHOICES,
        initial=ml.MODEL_SARIMAX,
        label="Model",
        widget=forms.Select(attrs={"class": "field field-select"}),
    )
    target_date = forms.DateField(
        label="Forecast date",
        widget=forms.DateInput(attrs={"type": "date", "class": "field field-date"}),
        error_messages={
            "required": "Pick a date to forecast.",
            "invalid": "That doesn't look like a valid date.",
        },
    )
    override_temp = forms.FloatField(
        label="Known temperature to start from (\u00b0C)",
        required=False,
        help_text=(
            "Optional — only used by Persistence. Leave blank to use the last "
            "recorded day in the dataset."
        ),
        widget=forms.NumberInput(attrs={"class": "field field-number", "step": "0.1", "placeholder": "e.g. 21.5"}),
        error_messages={"invalid": "Enter a plain number, e.g. 21.5."},
    )

    def clean(self):
        cleaned = super().clean()
        target_date = cleaned.get("target_date")
        override_temp = cleaned.get("override_temp")
        model_choice = cleaned.get("model_choice")

        if target_date is None:
            return cleaned

        engine = ml.get_engine()

        try:
            engine.validate_target_date(target_date)
        except ml.ForecastError as exc:
            self.add_error("target_date", str(exc))
        else:
            if model_choice == ml.MODEL_SEASONAL_NAIVE:
                try:
                    engine.seasonal_naive_lookup(target_date)
                except ml.ForecastError as exc:
                    self.add_error("target_date", str(exc))

        if override_temp is not None:
            try:
                engine.validate_override_temp(override_temp)
            except ml.ForecastError as exc:
                self.add_error("override_temp", str(exc))
            if model_choice and model_choice != ml.MODEL_PERSISTENCE:
                self.add_error(
                    "override_temp",
                    "This field only affects the Persistence model — clear it or switch models.",
                )

        return cleaned
