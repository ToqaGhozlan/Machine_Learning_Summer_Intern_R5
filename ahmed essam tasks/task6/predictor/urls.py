from django.urls import path
from .views import PredictorView, api_predict

urlpatterns = [
    path('', PredictorView.as_view(), name='predictor'),
    path('api/predict/', api_predict, name='api_predict'),
]
