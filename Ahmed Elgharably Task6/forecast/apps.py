import os
import sys

from django.apps import AppConfig


class ForecastConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "forecast"

    def ready(self):
        # `runserver`'s autoreloader starts a parent "watcher" process and a
        # child process that actually serves requests; RUN_MAIN is only set
        # in the child. Skip the watcher so the model is loaded exactly once
        # instead of twice. Any other command (gunicorn, migrate, test, ...)
        # loads it straight away since "runserver" won't be in argv at all.
        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return

        from . import ml

        try:
            ml.get_engine()
        except FileNotFoundError:
            # ml_artifacts/ not populated yet — the first real request will
            # trigger the lazy singleton in ml.get_engine() instead.
            pass
