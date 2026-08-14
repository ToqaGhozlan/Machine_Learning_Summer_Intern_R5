from django.apps import AppConfig


class ForecastConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'forecast'

    def ready(self):
        # Import triggers model_loader.py to run and load
        # the model, scaler, and config into memory once.
        from forecast.ml import model_loader