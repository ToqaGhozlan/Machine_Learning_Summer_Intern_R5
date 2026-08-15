import datetime as dt
import json
from pathlib import Path

from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .forms import ForecastDateForm
from .ml_model import get_model



def _load_evaluation():
    """Load precomputed chronological holdout metrics for the dashboard."""
    path = Path(__file__).resolve().parent.parent / "evaluation_results.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _cached_prediction(target_date: dt.date):
    """Bonus: cache repeated predictions so identical dates return instantly
    instead of re-running the SARIMAX forecast every time."""
    cache_key = f"forecast:{target_date.isoformat()}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached, True

    model = get_model()
    result = model.predict_for_date(target_date)
    cache.set(cache_key, result, timeout=60 * 60)  # 1 hour
    return result, False


def _temp_category(temp_c: float) -> dict:
    """Simple color/icon coding for the result panel."""
    if temp_c < 10:
        return {"label": "Cold", "css_class": "text-primary", "icon": "\u2744\ufe0f", "bg_class": "bg-primary-subtle"}
    if temp_c < 20:
        return {"label": "Mild", "css_class": "text-success", "icon": "\U0001F324\ufe0f", "bg_class": "bg-success-subtle"}
    if temp_c < 28:
        return {"label": "Warm", "css_class": "text-warning", "icon": "\u2600\ufe0f", "bg_class": "bg-warning-subtle"}
    return {"label": "Hot", "css_class": "text-danger", "icon": "\U0001F525", "bg_class": "bg-danger-subtle"}


@require_http_methods(["GET", "POST"])
def forecast_view(request):
    model = get_model()
    context = {
        "min_date": model.min_valid_date().isoformat(),
        "max_date": model.max_valid_date().isoformat(),
        "last_train_date": model.last_train_date.isoformat(),
        "first_train_date": model.first_train_date().isoformat(),
        "region": model.region,
        "model_version": model.model_version,
        "recent_history": model.recent_history(30),
        "recent_history_json": json.dumps(model.recent_history(30)),
        "evaluation": _load_evaluation(),
    }

    if request.method == "POST":
        form = ForecastDateForm(request.POST)
        if form.is_valid():
            target_date = form.cleaned_data["target_date"]
            try:
                result, from_cache = _cached_prediction(target_date)
                context["result"] = result
                context["from_cache"] = from_cache
                context["category"] = _temp_category(result.predicted_temp_c)
            except ValueError as exc:
                # Defensive: form validation already covers the range, but the
                # underlying model raises the same ValueError independently —
                # belt-and-suspenders so an inconsistency never surfaces as a 500.
                form.add_error("target_date", str(exc))
    else:
        form = ForecastDateForm()

    context["form"] = form
    return render(request, "predictor/forecast.html", context)


@require_http_methods(["GET"])
def forecast_api_view(request):
    """Bonus: JSON response option alongside the normal page, for other apps
    to consume. GET /api/forecast/?date=YYYY-MM-DD"""
    date_str = request.GET.get("date", "")
    try:
        target_date = dt.date.fromisoformat(date_str)
    except ValueError:
        return JsonResponse(
            {"error": f"Invalid or missing 'date' query parameter: {date_str!r}. Expected YYYY-MM-DD."},
            status=400,
        )

    model = get_model()
    if target_date < model.min_valid_date() or target_date > model.max_valid_date():
        return JsonResponse(
            {
                "error": "Date out of supported range.",
                "min_date": model.min_valid_date().isoformat(),
                "max_date": model.max_valid_date().isoformat(),
            },
            status=400,
        )

    result, from_cache = _cached_prediction(target_date)
    return JsonResponse({
        "region": model.region,
        "target_date": result.target_date.isoformat(),
        "days_ahead": result.days_ahead,
        "predicted_temp_c": result.predicted_temp_c,
        "confidence_interval_95": {"lower_c": result.ci_lower_c, "upper_c": result.ci_upper_c},
        "model_version": model.model_version,
        "from_cache": from_cache,
    })


@require_http_methods(["GET"])
def evaluation_api_view(request):
    """Return the precomputed holdout evaluation metrics as JSON."""
    evaluation = _load_evaluation()
    if evaluation is None:
        return JsonResponse({"error": "Evaluation results are not available. Run evaluate_model.py."}, status=503)
    return JsonResponse(evaluation)
