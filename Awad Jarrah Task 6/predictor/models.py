from django.db import models


class Prediction(models.Model):
    """A logged record of one prediction request, so we can show a
    'recent inputs / confidence' chart and browse history in the admin."""

    MODEL_CHOICES_LABELS = {
        "arma_manual": "Manual ARMA",
        "armax_auto": "Auto ARMAX",
        "fourier_auto": "Auto ARMAX + Fourier",
    }

    model_choice = models.CharField(max_length=20)
    predicted_temp = models.FloatField()
    lower = models.FloatField(help_text="Lower bound of the 95% confidence interval")
    upper = models.FloatField(help_text="Upper bound of the 95% confidence interval")
    from_cache = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        label = self.MODEL_CHOICES_LABELS.get(self.model_choice, self.model_choice)
        return f"{label} @ {self.created_at:%Y-%m-%d %H:%M} -> {self.predicted_temp:.1f}°C"
