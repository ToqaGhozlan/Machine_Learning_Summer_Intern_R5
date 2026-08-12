import dataclasses
from datetime import date as date_cls

from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from . import ml
from .forms import ForecastForm


def _cache_key(model_choice: str, target_date: date_cls, override_temp) -> str:
    return f"forecast:{model_choice}:{target_date.isoformat()}:{override_temp}"


def _temp_band(temp_c: float) -> str:
    """Cold / mild / warm bucket for the result panel's colour + icon,
    thresholds set from the dataset's own 12.4-30.8\u00b0C observed range."""
    if temp_c < 15:
        return "cold"
    if temp_c > 25:
        return "warm"
    return "mild"


def _run_prediction(model_choice: str, target_date: date_cls, override_temp):
    """Shared by the page view and the JSON API. Returns a plain dict
    (cache-friendly, JSON-serialisable) and caches it — repeated identical
    requests skip both the SARIMAX forecast call and the dataclass build."""
    key = _cache_key(model_choice, target_date, override_temp)
    cached = cache.get(key)
    if cached is not None:
        return cached, True

    engine = ml.get_engine()
    result = engine.predict(model_choice, target_date, override_temp)
    result_dict = dataclasses.asdict(result)
    result_dict["target_date"] = result.target_date.isoformat()
    result_dict["temp_band"] = _temp_band(result.predicted_temp_c)

    cache.set(key, result_dict, settings.PREDICTION_CACHE_TTL_SECONDS)
    return result_dict, False


@require_http_methods(["GET", "POST"])
def index(request):
    engine = ml.get_engine()
    result = None
    from_cache = False

    if request.method == "POST":
        form = ForecastForm(request.POST)
        if form.is_valid():
            result, from_cache = _run_prediction(
                form.cleaned_data["model_choice"],
                form.cleaned_data["target_date"],
                form.cleaned_data.get("override_temp"),
            )
    else:
        form = ForecastForm(initial={"model_choice": ml.MODEL_SARIMAX})

    bounds = engine.date_bounds()
    form.fields["target_date"].widget.attrs.update({"min": bounds["min"], "max": bounds["max"]})

    context = {
        "form": form,
        "result": result,
        "from_cache": from_cache,
        "date_bounds": bounds,
        "recent_history": engine.recent_history(30),
        "meta": engine.meta,
    }
    return render(request, "forecast/index.html", context)


@require_http_methods(["GET"])
def api_predict(request):
    """Bonus: JSON response alongside the normal page, for other apps to use.

    GET /api/predict/?model=sarimax&date=2026-03-15&override_temp=21.5
    """
    model_choice = request.GET.get("model", ml.MODEL_SARIMAX)
    date_str = request.GET.get("date")
    override_str = request.GET.get("override_temp")

    if not date_str:
        return JsonResponse({"error": "Missing required query param 'date' (YYYY-MM-DD)."}, status=400)

    try:
        target_date = date_cls.fromisoformat(date_str)
    except ValueError:
        return JsonResponse({"error": f"'{date_str}' is not a valid YYYY-MM-DD date."}, status=400)

    override_temp = None
    if override_str is not None:
        try:
            override_temp = float(override_str)
        except ValueError:
            return JsonResponse({"error": f"'{override_str}' is not a valid number for override_temp."}, status=400)

    if model_choice not in dict(ml.MODEL_CHOICES):
        return JsonResponse(
            {"error": f"'{model_choice}' is not one of {list(dict(ml.MODEL_CHOICES))}."}, status=400
        )

    try:
        result_dict, from_cache = _run_prediction(model_choice, target_date, override_temp)
    except ml.ForecastError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    result_dict["from_cache"] = from_cache
    return JsonResponse(result_dict)
