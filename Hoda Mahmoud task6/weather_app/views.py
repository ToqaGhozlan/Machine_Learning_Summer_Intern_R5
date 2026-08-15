"""
Django views for weather prediction.
"""

from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
import logging

from .forms import TemperaturePredictionForm
from .ml_model import predict_temperature, get_model_info

logger = logging.getLogger(__name__)


@require_http_methods(["GET", "POST"])
@csrf_protect
def predict_view(request):
    """
    Main view for temperature prediction.
    
    Handles both GET (display form) and POST (process prediction) requests.
    """
    prediction_result = None
    form_errors = None
    model_info = get_model_info()
    
    if request.method == 'POST':
        form = TemperaturePredictionForm(request.POST)
        
        if form.is_valid():
            horizon_days = form.cleaned_data['horizon_days']
            
            # Get prediction from ML model
            prediction_result = predict_temperature(horizon_days)
            
            if not prediction_result.get('success'):
                form_errors = prediction_result.get('error', 'Unknown error occurred')
        else:
            form_errors = 'Invalid form submission. Please try again.'
    else:
        form = TemperaturePredictionForm()
    
    context = {
        'form': form,
        'prediction_result': prediction_result,
        'form_errors': form_errors,
        'model_info': model_info,
    }
    
    return render(request, 'weather_app/prediction.html', context)


def info_view(request):
    """
    View displaying model information.
    """
    model_info = get_model_info()
    
    context = {
        'model_info': model_info,
    }
    
    return render(request, 'weather_app/info.html', context)
