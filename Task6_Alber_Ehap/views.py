from django.shortcuts import render

from .forms import WeatherPredictionForm

from .model_service import predict_weather


def home(request):

    prediction = None


    form = WeatherPredictionForm(
        request.POST or None
    )


    if request.method == "POST":

        if form.is_valid():

            try:

                prediction = predict_weather(
                    form.cleaned_data
                )

            except Exception as error:

                form.add_error(
                    None,
                    f"Prediction failed: {error}"
                )


    return render(
        request,

        "predictor/index.html",

        {
            "form": form,
            "prediction": prediction,
        }
    )