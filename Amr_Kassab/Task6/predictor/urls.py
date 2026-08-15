from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path('', views.predict_view, name='predict'),
    path('api/predict/', views.api_predict_view, name='api_predict'),
]
