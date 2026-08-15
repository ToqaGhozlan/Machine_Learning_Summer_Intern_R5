import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .forms import WeatherPredictionForm
from .ml_service import predict_temperature, get_model_metrics, get_dataset_info


def predict_view(request):
    prediction = None
    form = WeatherPredictionForm()

    if request.method == 'POST':
        form = WeatherPredictionForm(request.POST)
        if form.is_valid():
            input_data = {
                'relative_humidity': form.cleaned_data['relative_humidity'],
                'precipitation': form.cleaned_data['precipitation'],
                'wind_speed': form.cleaned_data['wind_speed'],
                'cloud_cover': form.cleaned_data['cloud_cover'],
                'surface_pressure': form.cleaned_data['surface_pressure'],
                'hour': form.cleaned_data['hour'],
                'month': form.cleaned_data['month'],
            }
            model_name = form.cleaned_data.get('model_choice', 'gradient_boosting') or 'gradient_boosting'

            try:
                prediction = predict_temperature(input_data, model_name)
                prediction['input_data'] = input_data
            except Exception as e:
                prediction = {'error': str(e)}

    model_metrics = get_model_metrics()
    dataset_info = get_dataset_info()

    context = {
        'form': form,
        'prediction': prediction,
        'model_metrics': model_metrics,
        'dataset_info': dataset_info,
    }
    return render(request, 'predictor/predict.html', context)


@csrf_exempt
def api_predict_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    required_fields = ['relative_humidity', 'precipitation', 'wind_speed',
                       'cloud_cover', 'surface_pressure', 'hour', 'month']
    missing = [f for f in required_fields if f not in data]
    if missing:
        return JsonResponse({'error': f'Missing fields: {", ".join(missing)}'}, status=400)

    model_name = data.get('model', 'gradient_boosting')

    try:
        result = predict_temperature(data, model_name)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
