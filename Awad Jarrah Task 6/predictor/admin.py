from django.contrib import admin

from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("created_at", "model_choice", "predicted_temp", "lower", "upper", "from_cache")
    list_filter = ("model_choice", "from_cache")
    ordering = ("-created_at",)
    readonly_fields = ("model_choice", "predicted_temp", "lower", "upper", "from_cache", "created_at")
