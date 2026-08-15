from django.shortcuts import render
import traceback

from .forms import ForecastDateForm
from .ml_model import predict_for_date


def predict_view(request):
    result = None
    error = None

    if request.method == "POST":
        form = ForecastDateForm(request.POST)
        if form.is_valid():
            try:
                result = predict_for_date(form.cleaned_data["forecast_date"])
            except FileNotFoundError:
                error = "Model file not found on the server. Check models/sarima_fourier_model.pkl."
                traceback.print_exc()
            except ValueError as e:
                error = str(e)
            except Exception:
                error = "Something went wrong while generating the forecast. Please try again."
                traceback.print_exc()
    else:
        form = ForecastDateForm()

    return render(request, "weather/index.html", {
        "form": form,
        "result": result,
        "error": error,
    })
