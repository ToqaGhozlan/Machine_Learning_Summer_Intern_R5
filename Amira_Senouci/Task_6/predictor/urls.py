from django.urls import path

from . import views

app_name = "predictor"

urlpatterns = [
    path("", views.forecast_view, name="forecast"),
    path("api/forecast/", views.forecast_api_view, name="forecast_api"),
    path("api/evaluation/", views.evaluation_api_view, name="evaluation_api"),
]
