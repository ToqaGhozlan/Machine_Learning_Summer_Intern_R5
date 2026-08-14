from django.urls import path
from forecast.views import PredictNextDayView

urlpatterns = [
    path("predict/", PredictNextDayView.as_view(), name="predict_next_day"),
]