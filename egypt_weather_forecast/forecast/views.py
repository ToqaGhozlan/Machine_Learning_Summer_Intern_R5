import math
from datetime import timedelta, datetime

from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
from forecast.models import WeatherRecord
from forecast.ml.predictor import predict_next_day
from forecast.ml.model_loader import LOOKBACK


def get_allowed_date_range():
    """Earliest/latest dates for which a prediction can be made, given LOOKBACK."""
    earliest = WeatherRecord.objects.order_by("date").values_list("date", flat=True).first()
    latest = WeatherRecord.objects.order_by("-date").values_list("date", flat=True).first()
    if not earliest or not latest:
        return None, None
    min_date = earliest + timedelta(days=LOOKBACK)
    max_date = latest + timedelta(days=1)
    return min_date, max_date


def get_prediction_for_date(target_date=None):
    """
    Predict T2M for target_date using the LOOKBACK days immediately before it.
    If target_date is None, predicts the day after the latest stored record.
    Returns None if there isn't enough history before target_date.
    """
    if target_date is None:
        records = WeatherRecord.objects.order_by("-date")[:LOOKBACK]
        records = list(records)[::-1]
        if len(records) < LOOKBACK:
            return None
        last_date = records[-1].date
        target_date = last_date + timedelta(days=1)
    else:
        records = WeatherRecord.objects.filter(date__lt=target_date).order_by("-date")[:LOOKBACK]
        records = list(records)[::-1]
        if len(records) < LOOKBACK:
            return None
        last_date = records[-1].date

    recent_values = [r.t2m for r in records]
    prediction = predict_next_day(recent_values)

    return {
        "predicted_date": target_date,
        "predicted_t2m_celsius": round(prediction, 2),
        "based_on_last_date": last_date,
    }


class PredictNextDayView(View):
    def get(self, request):
        result = get_prediction_for_date()
        if result is None:
            return JsonResponse(
                {"error": f"Not enough history. Need {LOOKBACK} days."},
                status=400
            )
        return JsonResponse({
            "predicted_date": str(result["predicted_date"]),
            "predicted_t2m_celsius": result["predicted_t2m_celsius"],
            "based_on_last_date": str(result["based_on_last_date"]),
        })


class DashboardView(View):
    """Renders the sundial console dashboard: gauge + N-day sparkline."""
    SPARKLINE_DAYS = 60

    def get(self, request):
        min_date, max_date = get_allowed_date_range()
        date_param = request.GET.get("date", "").strip()
        error = None
        target_date = None

        if date_param:
            try:
                target_date = datetime.strptime(date_param, "%Y-%m-%d").date()
            except ValueError:
                error = "Enter a valid date (YYYY-MM-DD)."
            else:
                if min_date and max_date and not (min_date <= target_date <= max_date):
                    error = f"Date must be between {min_date} and {max_date}."
                    target_date = None

        prediction = None if error else get_prediction_for_date(target_date)
        if prediction is None and not error:
            error = "Not enough history to forecast that date."

        anchor_date = prediction["based_on_last_date"] if prediction else max_date
        sparkline_records = (
            WeatherRecord.objects
            .filter(date__lte=anchor_date)
            .order_by("-date")[:self.SPARKLINE_DAYS]
            if anchor_date else []
        )
        sparkline_records = list(sparkline_records)[::-1]
        sparkline_data = [
            {"date": str(r.date), "t2m": r.t2m} for r in sparkline_records
        ]

        temps = [d["t2m"] for d in sparkline_data]
        if prediction:
            temps.append(prediction["predicted_t2m_celsius"])

        gauge_min = math.floor(min(temps)) - 2 if temps else 0
        gauge_max = math.ceil(max(temps)) + 2 if temps else 40

        context = {
            "prediction": prediction,
            "sparkline_data": sparkline_data,
            "gauge_min": gauge_min,
            "gauge_max": gauge_max,
            "min_date": min_date,
            "max_date": max_date,
            "selected_date": date_param,
            "error": error,
        }
        return render(request, "forecast/index.html", context)