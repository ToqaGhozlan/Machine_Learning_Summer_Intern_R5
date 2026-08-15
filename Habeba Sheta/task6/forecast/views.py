from django.shortcuts import render
from .forms import WeatherForecastForm, MIN_ALLOWED_DATE, MAX_ALLOWED_DATE
from .predictor import predict_temperature, last_known_date

# أقل وأعلى حرارة سُجلت فعليًا في بيانات القاهرة لرسم شريط المدى الحراري
DATASET_MIN_TEMP = 6.67
DATASET_MAX_TEMP = 35.05

ENGLISH_WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ENGLISH_MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']


def _percent_on_scale(value):
    """بتحول درجة الحرارة لنسبة مئوية (0-100) على شريط المدى، عشان نرسم بيها في CSS"""
    pct = (value - DATASET_MIN_TEMP) / (DATASET_MAX_TEMP - DATASET_MIN_TEMP) * 100
    return max(0, min(100, round(pct, 1)))


def _format_english_date(d):
    weekday = ENGLISH_WEEKDAYS[d.weekday()]
    month = ENGLISH_MONTHS[d.month - 1]
    return f"{weekday}, {month} {d.day}, {d.year}"


def forecast_view(request):
    result = None

    if request.method == 'POST':
        form = WeatherForecastForm(request.POST)
        if form.is_valid():
            target_date = form.cleaned_data['target_date']
            predicted, ci_low, ci_high, steps_ahead = predict_temperature(
                __import__('pandas').Timestamp(target_date)
            )

            if predicted < 15:
                category, category_label = 'cold', 'Cold weather'
            elif predicted > 28:
                category, category_label = 'hot', 'Hot weather'
            else:
                category, category_label = 'mild', 'Mild weather'

            ci_low_pct = _percent_on_scale(ci_low)
            ci_high_pct = _percent_on_scale(ci_high)

            result = {
                'target_date_display': _format_english_date(target_date),
                'predicted': predicted,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'steps_ahead': steps_ahead,
                'category': category,
                'category_label': category_label,
                'predicted_pct': _percent_on_scale(predicted),
                'ci_low_pct': ci_low_pct,
                'ci_width_pct': round(ci_high_pct - ci_low_pct, 1),
            }
    else:
        form = WeatherForecastForm()

    return render(request, 'forecast/index.html', {
        'form': form,
        'result': result,
        'last_known_date': last_known_date,
        'min_date': MIN_ALLOWED_DATE,
        'max_date': MAX_ALLOWED_DATE,
        'dataset_min': DATASET_MIN_TEMP,
        'dataset_max': DATASET_MAX_TEMP,
    })