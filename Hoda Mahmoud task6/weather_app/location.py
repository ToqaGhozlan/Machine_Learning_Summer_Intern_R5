"""Weather-dataset location metadata used by the Django presentation layer.

Coordinates come from the saved NASA POWER response that produced the deployed
model, not the legacy ``outputs/Cairo`` directory name.  The small mapping
avoids a runtime reverse-geocoding dependency.
"""

import json
from pathlib import Path

from django.conf import settings


# NASA POWER returned its resolved grid point (51.507, -0.128); the Task 5
# summary records the requested point with greater precision (51.5074, -0.1278).
KNOWN_LOCATIONS = (((51.5074, -0.1278), "London", "United Kingdom"),)
COORDINATE_TOLERANCE = 0.01


def _format_coordinate(value, positive_suffix, negative_suffix):
    suffix = positive_suffix if value >= 0 else negative_suffix
    return f"{abs(value):.2f}\N{DEGREE SIGN}{suffix}"


def get_weather_location():
    """Return display metadata from the persisted NASA POWER dataset source."""
    try:
        with Path(settings.WEATHER_DATA_RESPONSE_PATH).open(encoding="utf-8") as response_file:
            longitude, latitude, *_ = json.load(response_file)["geometry"]["coordinates"]
    except (FileNotFoundError, IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {"name": "Saved weather dataset", "full_name": "Saved weather dataset", "coordinates": "coordinates unavailable"}

    for (known_latitude, known_longitude), name, country in KNOWN_LOCATIONS:
        if abs(latitude - known_latitude) <= COORDINATE_TOLERANCE and abs(longitude - known_longitude) <= COORDINATE_TOLERANCE:
            full_name = f"{name}, {country}"
            break
    else:
        name, full_name = "Saved weather dataset", "Saved weather dataset"

    return {
        "name": name,
        "full_name": full_name,
        "coordinates": f"approximately {_format_coordinate(latitude, 'N', 'S')}, {_format_coordinate(longitude, 'E', 'W')}",
        "latitude": latitude,
        "longitude": longitude,
    }


def weather_location(request):
    """Make the dataset location available to every Django template."""
    return {"weather_location": get_weather_location()}
