from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictor"

    def ready(self):
        # Warm up / load the pickled models once, when the Django app starts,
        # so the first request isn't slow and every request reuses the same
        # in-memory model objects instead of unpickling on every submit.
        from . import ml_models

        ml_models.load_all_models()
