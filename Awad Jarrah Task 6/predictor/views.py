import hashlib
import json

from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render

from . import ml_models
from .forms import WeatherPredictionForm
from .models import Prediction

RECENT_LIMIT = 10


def _cache_key(cleaned_data):
    """Stable cache key from the submitted (validated) form values, so an
    identical submission returns instantly instead of re-running the model."""
    payload = json.dumps(cleaned_data, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"prediction:{digest}"


def _run_prediction(cleaned_data):
    """Shared by the HTML view and the JSON API: checks the cache, else
    calls the model and caches the result for 1 hour. Every call (cached or
    not) is logged to the Prediction table for the recent-activity chart."""
    key = _cache_key(cleaned_data)
    cached = cache.get(key)

    if cached is not None:
        result, from_cache = cached, True
    else:
        result = ml_models.predict(cleaned_data["model_choice"], cleaned_data)
        cache.set(key, result, timeout=60 * 60)
        from_cache = False

    Prediction.objects.create(
        model_choice=cleaned_data["model_choice"],
        predicted_temp=result["predicted_temp"],
        lower=result["lower"],
        upper=result["upper"],
        from_cache=from_cache,
    )

    return result, from_cache


def _recent_chart_data():
    """Last RECENT_LIMIT predictions, oldest -> newest, packaged for Chart.js."""
    recent = list(Prediction.objects.order_by("-created_at")[:RECENT_LIMIT])
    recent.reverse()  # chronological order for the chart

    return {
        "labels": [p.created_at.strftime("%m-%d %H:%M") for p in recent],
        "predicted": [round(p.predicted_temp, 2) for p in recent],
        "lower": [round(p.lower, 2) for p in recent],
        "upper": [round(p.upper, 2) for p in recent],
        "model_choice": [p.model_choice for p in recent],
    }


def predict_view(request):
    result = None
    from_cache = False
    load_errors = ml_models.get_load_errors()

    if request.method == "POST":
        form = WeatherPredictionForm(request.POST)
        if form.is_valid():
            try:
                result, from_cache = _run_prediction(form.cleaned_data)
            except ValueError as exc:
                form.add_error(None, str(exc))
    else:
        form = WeatherPredictionForm()

    return render(
        request,
        "predictor/index.html",
        {
            "form": form,
            "result": result,
            "from_cache": from_cache,
            "load_errors": load_errors,
            "exog_fields": WeatherPredictionForm.EXOG_FIELDS,
            "chart_data": _recent_chart_data(),
        },
    )


def predict_json(request):
    """Bonus: same prediction logic, but returns JSON for other apps to consume.

    Accepts the same fields as the HTML form via GET or POST.
    Example: /api/predict/?model_choice=arma_manual&forecast_date=2025-08-05
    """
    data = request.POST if request.method == "POST" else request.GET
    form = WeatherPredictionForm(data)

    if not form.is_valid():
        return JsonResponse({"ok": False, "errors": form.errors}, status=400)

    try:
        result, from_cache = _run_prediction(form.cleaned_data)
    except ValueError as exc:
        return JsonResponse({"ok": False, "errors": {"__all__": [str(exc)]}}, status=400)

    return JsonResponse({"ok": True, "cached": from_cache, "result": result})
