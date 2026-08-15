"""
Django app configuration for weather_app.
"""

from django.apps import AppConfig


class WeatherAppConfig(AppConfig):
    """Configuration class for weather_app"""
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'weather_app'
    verbose_name = 'Weather Prediction App'
    
    def ready(self):
        """Initialize app when Django starts"""
        # Load the model when app is ready
        from .ml_model import load_model
        load_model()
