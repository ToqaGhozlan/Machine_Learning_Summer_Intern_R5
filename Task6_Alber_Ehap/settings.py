from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


SECRET_KEY = "django-task6-development-key"

DEBUG = True

ALLOWED_HOSTS = [
    "127.0.0.1",
    "localhost",
]


INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",

    "predictor",
]


MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]


ROOT_URLCONF = "weather_project.urls"


TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",

        "DIRS": [],

        "APP_DIRS": True,

        "OPTIONS": {
            "context_processors": [],
        },
    },
]


WSGI_APPLICATION = "weather_project.wsgi.application"


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


LANGUAGE_CODE = "en-us"

TIME_ZONE = "Africa/Cairo"

USE_I18N = True

USE_TZ = True


STATIC_URL = "static/"

STATIC_ROOT = BASE_DIR / "staticfiles"


DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"