from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'
    verbose_name = 'Weather Prediction'

    def ready(self):
        from . import ml_service
        ml_service._load_model('gradient_boosting')
        ml_service._load_scaler()
        ml_service._load_metadata()
