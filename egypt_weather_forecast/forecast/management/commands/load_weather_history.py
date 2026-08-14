import os
import pandas as pd
from django.core.management.base import BaseCommand
from django.conf import settings
from forecast.models import WeatherRecord


class Command(BaseCommand):
    help = "Load historical Egypt-wide daily T2M data into WeatherRecord table"

    def handle(self, *args, **options):
        csv_path = os.path.join(settings.BASE_DIR, "data", "egypt_weather_2015_2024_raw.csv")

        self.stdout.write(f"Reading {csv_path} ...")
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])

        # Egypt-wide daily average, same as in the notebook
        egypt_t2m = (
            df.groupby("Date")["T2M"]
              .mean()
              .sort_index()
        )

        self.stdout.write(f"Prepared {len(egypt_t2m)} daily records. Inserting...")

        records = [
            WeatherRecord(date=date.date(), t2m=float(value))
            for date, value in egypt_t2m.items()
        ]

        # update_or_create-style bulk insert: skip duplicates safely
        WeatherRecord.objects.bulk_create(
            records,
            ignore_conflicts=True  # skips rows that violate the unique date constraint
        )

        self.stdout.write(self.style.SUCCESS(
            f"Done. WeatherRecord table now has {WeatherRecord.objects.count()} rows."
        ))