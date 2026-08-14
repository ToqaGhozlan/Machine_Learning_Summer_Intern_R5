from django.shortcuts import render
import traceback
 
from .forms import WeatherInputForm
from .ml_model import predict_next_day, WEATHER_FEATURES
 
 
def predict_view(request):
    result = None
    error = None
 
    if request.method == "POST":
        form = WeatherInputForm(request.POST)
        if form.is_valid():
            weather_values = {f: form.cleaned_data[f] for f in WEATHER_FEATURES}
            try:
                result = predict_next_day(weather_values)# func to predict the next day temperature based on the user input weather values
            except FileNotFoundError:
                error = "Model file not found on the server. Check models/sarima_fourier_model.pkl."
                traceback.print_exc()
            except Exception:
                error = "Something went wrong while generating the forecast. Please try again."
                traceback.print_exc()
    else:
        form = WeatherInputForm()
 
    return render(request, "weather/index.html", {
        "form": form,
        "result": result,
        "error": error,
    })