from django.urls import path

from . import views

app_name = "weather"

urlpatterns = [
    path("", views.predict_view, name="predict"),
]
