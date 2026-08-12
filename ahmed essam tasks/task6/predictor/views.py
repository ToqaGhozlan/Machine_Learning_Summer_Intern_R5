from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
import json
from .forms import WeatherPredictionForm
from .model_loader import pipeline

class PredictorView(View):
    template_name = 'predictor/index.html'
    
    def get(self, request):
        # Load default data to pre-fill the form
        default_data = pipeline.get_default_data()
        
        form = WeatherPredictionForm(initial=default_data)
        
        context = {
            'form': form,
            'default_data': json.dumps(default_data)
        }
        return render(request, self.template_name, context)
        
    def post(self, request):
        form = WeatherPredictionForm(request.POST)
        context = {'form': form}
        
        if form.is_valid():
            try:
                # Pass the cleaned data dictionary directly
                prediction = pipeline.predict(form.cleaned_data)
                context['prediction'] = round(prediction, 2)
                context['input_data'] = json.dumps(form.cleaned_data)
                
                # Determine context message (e.g. cold, warm, hot)
                if prediction < 15:
                    context['temp_category'] = 'cold'
                    context['message'] = "It's looking chilly tomorrow."
                elif prediction < 25:
                    context['temp_category'] = 'mild'
                    context['message'] = "Mild and comfortable weather ahead."
                elif prediction < 35:
                    context['temp_category'] = 'warm'
                    context['message'] = "It's going to be warm tomorrow."
                else:
                    context['temp_category'] = 'hot'
                    context['message'] = "Hot weather expected! Stay hydrated."
                    
            except Exception as e:
                context['error'] = f"Prediction failed: {str(e)}"
        
        # Pass the default data in case they want to reset
        context['default_data'] = json.dumps(pipeline.get_default_data())
        
        return render(request, self.template_name, context)

def api_predict(request):
    """API endpoint for predictions"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Basic validation
            required_fields = ['temp_mean', 'humidity', 'wind_speed', 'precipitation']
            if not all(field in data for field in required_fields):
                return JsonResponse({'error': f'Missing required fields. Expected: {required_fields}'}, status=400)
                
            prediction = pipeline.predict(data)
            return JsonResponse({'prediction': round(prediction, 2)})
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
