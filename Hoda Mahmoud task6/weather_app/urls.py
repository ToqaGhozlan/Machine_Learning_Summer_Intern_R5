"""
URL configuration for weather_app.
"""

from django.urls import path
from . import views

app_name = 'weather_app'

urlpatterns = [
    path('', views.predict_view, name='predict'),
    path('info/', views.info_view, name='info'),
]
