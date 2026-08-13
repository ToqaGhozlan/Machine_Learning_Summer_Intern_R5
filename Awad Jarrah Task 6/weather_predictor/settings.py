"""
Django settings for weather_predictor project.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# SECURITY
# --------------------------------------------------------------------------
# Do NOT hardcode a real secret key in production. This is fine for local
# dev / a class assignment; before deploying publicly, set this via an
# environment variable instead.
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "dev-only-insecure-secret-key-change-me",
)

# Turn this OFF before deploying publicly (see README).
DEBUG = os.environ.get("DJANGO_DEBUG", "True") == "True"

ALLOWED_HOSTS = os.environ.get(
    "DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost"
).split(",")

# --------------------------------------------------------------------------
# APPLICATIONS
# --------------------------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "predictor",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "weather_predictor.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "weather_predictor.wsgi.application"

# --------------------------------------------------------------------------
# DATABASE (not used for predictions, but Django needs one for admin/sessions)
# --------------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "predictor" / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Simple in-memory cache so identical inputs return instantly on repeat
# submissions (bonus challenge) without needing Redis/Memcached for a demo.
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "weather-predictor-cache",
    }
}

# --------------------------------------------------------------------------
# ML MODEL CONFIG (custom setting used by predictor/ml_models.py)
# --------------------------------------------------------------------------
# Folder where the three pickled models from Task 5 live.
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

# First day of the training set used to fit the models (from Task 5 notebook).
# Needed to compute the correct annual Fourier terms for a given forecast date.
TRAIN_START_DATE = "2019-01-01"
TRAIN_END_DATE = "2025-07-31"
