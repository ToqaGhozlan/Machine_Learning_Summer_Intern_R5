from django.db import models


class WeatherRecord(models.Model):
    date = models.DateField(unique=True)
    t2m = models.FloatField(help_text="Daily average temperature at 2m (°C)")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["date"]

    def __str__(self):
        return f"{self.date} - {self.t2m}°C"