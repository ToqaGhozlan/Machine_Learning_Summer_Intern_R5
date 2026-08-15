import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "weather_deploy.settings")
django.setup()

from django.test import Client
from django.test.utils import setup_test_environment
import datetime as dt

setup_test_environment()  # enables response.context capturing outside manage.py test
client = Client()

print("=" * 60)
print("TEST 1: GET / (initial form page)")
print("=" * 60)
resp = client.get("/")
print(f"Status: {resp.status_code}")
assert resp.status_code == 200
content = resp.content.decode()
assert "Algiers Weather Forecaster" in content
assert "Get Forecast" in content
print("PASS: page renders with title and form button")

print()
print("=" * 60)
print("TEST 2: POST valid date (within range)")
print("=" * 60)
resp = client.get("/")
min_date = resp.context["min_date"]
print(f"Model min valid date: {min_date}")
resp = client.post("/", {"target_date": min_date})
print(f"Status: {resp.status_code}")
assert resp.status_code == 200
assert "result" in resp.context
result = resp.context["result"]
print(f"Predicted temp: {result.predicted_temp_c}C  CI: [{result.ci_lower_c}, {result.ci_upper_c}]")
assert isinstance(result.predicted_temp_c, float)
print("PASS: valid prediction returned")

print()
print("=" * 60)
print("TEST 3: POST empty field (missing input)")
print("=" * 60)
resp = client.post("/", {"target_date": ""})
print(f"Status: {resp.status_code}")
assert resp.status_code == 200
assert "result" not in resp.context or resp.context.get("result") is None
assert resp.context["form"].errors
print(f"Form errors: {resp.context['form'].errors}")
print("PASS: empty field rejected with form error, no 500")

print()
print("=" * 60)
print("TEST 4: POST non-date text (invalid format)")
print("=" * 60)
resp = client.post("/", {"target_date": "not-a-date"})
print(f"Status: {resp.status_code}")
assert resp.status_code == 200
assert resp.context["form"].errors
print(f"Form errors: {resp.context['form'].errors}")
print("PASS: garbage input rejected cleanly")

print()
print("=" * 60)
print("TEST 5: POST out-of-range date (too far in the past)")
print("=" * 60)
resp = client.post("/", {"target_date": "2020-01-01"})
print(f"Status: {resp.status_code}")
assert resp.status_code == 200
assert resp.context["form"].errors
print(f"Form errors: {resp.context['form'].errors}")
print("PASS: past date rejected with clear message")

print()
print("=" * 60)
print("TEST 6: POST out-of-range date (too far in the future)")
print("=" * 60)
resp = client.get("/")
max_date = resp.context["max_date"]
too_far = (dt.date.fromisoformat(max_date) + dt.timedelta(days=30)).isoformat()
resp = client.post("/", {"target_date": too_far})
print(f"Status: {resp.status_code}")
assert resp.status_code == 200
assert resp.context["form"].errors
print(f"Form errors: {resp.context['form'].errors}")
print("PASS: far-future date rejected with clear message")

print()
print("=" * 60)
print("TEST 7: JSON API - valid request")
print("=" * 60)
resp = client.get(f"/api/forecast/?date={min_date}")
print(f"Status: {resp.status_code}")
assert resp.status_code == 200
data = resp.json()
print(data)
assert "predicted_temp_c" in data
print("PASS: API returns valid JSON with prediction")

print()
print("=" * 60)
print("TEST 8: JSON API - caching (second identical request)")
print("=" * 60)
resp1 = client.get(f"/api/forecast/?date={min_date}")
resp2 = client.get(f"/api/forecast/?date={min_date}")
print(f"First call from_cache: {resp1.json()['from_cache']}")
print(f"Second call from_cache: {resp2.json()['from_cache']}")
assert resp2.json()["from_cache"] is True
print("PASS: second identical request served from cache")

print()
print("=" * 60)
print("TEST 9: JSON API - invalid date param")
print("=" * 60)
resp = client.get("/api/forecast/?date=garbage")
print(f"Status: {resp.status_code}")
assert resp.status_code == 400
print(resp.json())
print("PASS: invalid API param returns 400, not a crash")

print()
print("=" * 60)
print("TEST 10: Cross-check prediction matches notebook-style direct call")
print("=" * 60)
from predictor.ml_model import get_model
model = get_model()
target = dt.date.fromisoformat(min_date)
direct_result = model.predict_for_date(target)
resp = client.post("/", {"target_date": min_date})
view_result = resp.context["result"]
print(f"Direct model call:  {direct_result.predicted_temp_c}")
print(f"Via Django view:     {view_result.predicted_temp_c}")
assert direct_result.predicted_temp_c == view_result.predicted_temp_c
print("PASS: view's prediction matches calling the model directly")

print()
print("ALL TESTS PASSED")
