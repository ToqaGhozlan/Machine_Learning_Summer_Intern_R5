from django.urls import path

from . import views

app_name = "predictor"

urlpatterns = [
    path("", views.predict_view, name="predict"),
    path("api/predict/", views.predict_json, name="predict_json"),
]
