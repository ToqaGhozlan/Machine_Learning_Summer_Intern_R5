import json

from django.test import TestCase
from django.urls import reverse

from . import ml


class IndexPageTests(TestCase):
    def test_get_renders_empty_form(self):
        resp = self.client.get(reverse("forecast:index"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Get a forecast")
        self.assertIsNone(resp.context["result"])

    def test_sarimax_valid_date_returns_prediction(self):
        resp = self.client.post(
            reverse("forecast:index"),
            {"model_choice": ml.MODEL_SARIMAX, "target_date": "2026-03-15", "override_temp": ""},
        )
        self.assertEqual(resp.status_code, 200)
        result = resp.context["result"]
        self.assertIsNotNone(result)
        self.assertEqual(result["model_choice"], ml.MODEL_SARIMAX)
        self.assertEqual(result["horizon_days"], 74)
        self.assertIsNotNone(result["ci_low_c"])
        self.assertLess(result["ci_low_c"], result["predicted_temp_c"])
        self.assertGreater(result["ci_high_c"], result["predicted_temp_c"])

    def test_persistence_uses_override_temp(self):
        resp = self.client.post(
            reverse("forecast:index"),
            {"model_choice": ml.MODEL_PERSISTENCE, "target_date": "2026-01-05", "override_temp": "19.5"},
        )
        result = resp.context["result"]
        self.assertEqual(result["predicted_temp_c"], 19.5)

    def test_persistence_defaults_to_last_known_value_without_override(self):
        engine = ml.get_engine()
        expected = round(float(engine.history.iloc[-1]), 1)
        resp = self.client.post(
            reverse("forecast:index"),
            {"model_choice": ml.MODEL_PERSISTENCE, "target_date": "2026-01-05", "override_temp": ""},
        )
        result = resp.context["result"]
        self.assertEqual(result["predicted_temp_c"], expected)

    def test_seasonal_naive_matches_historical_value(self):
        import datetime

        engine = ml.get_engine()
        target = "2026-06-10"
        expected = round(engine.seasonal_naive_lookup(datetime.date(2026, 6, 10)), 1)
        resp = self.client.post(
            reverse("forecast:index"),
            {"model_choice": ml.MODEL_SEASONAL_NAIVE, "target_date": target, "override_temp": ""},
        )
        result = resp.context["result"]
        self.assertEqual(result["predicted_temp_c"], expected)

    def test_date_before_last_training_date_is_rejected(self):
        resp = self.client.post(
            reverse("forecast:index"),
            {"model_choice": ml.MODEL_SARIMAX, "target_date": "2025-06-01", "override_temp": ""},
        )
        self.assertIsNone(resp.context["result"])
        self.assertTrue(resp.context["form"].errors.get("target_date"))

    def test_date_too_far_ahead_is_rejected(self):
        resp = self.client.post(
            reverse("forecast:index"),
            {"model_choice": ml.MODEL_SARIMAX, "target_date": "2028-01-01", "override_temp": ""},
        )
        self.assertIsNone(resp.context["result"])
        self.assertTrue(resp.context["form"].errors.get("target_date"))

    def test_missing_date_shows_required_error(self):
        resp = self.client.post(
            reverse("forecast:index"),
            {"model_choice": ml.MODEL_SARIMAX, "target_date": "", "override_temp": ""},
        )
        self.assertTrue(resp.context["form"].errors.get("target_date"))

    def test_non_numeric_override_temp_shows_error(self):
        resp = self.client.post(
            reverse("forecast:index"),
            {"model_choice": ml.MODEL_PERSISTENCE, "target_date": "2026-01-05", "override_temp": "warm-ish"},
        )
        self.assertTrue(resp.context["form"].errors.get("override_temp"))

    def test_out_of_range_override_temp_shows_error(self):
        resp = self.client.post(
            reverse("forecast:index"),
            {"model_choice": ml.MODEL_PERSISTENCE, "target_date": "2026-01-05", "override_temp": "999"},
        )
        self.assertTrue(resp.context["form"].errors.get("override_temp"))

    def test_seasonal_naive_out_of_lookup_range_shows_error(self):
        # 2027 dates have no matching day 365 days earlier in the 2024-2025 record.
        resp = self.client.post(
            reverse("forecast:index"),
            {"model_choice": ml.MODEL_SEASONAL_NAIVE, "target_date": "2027-02-01", "override_temp": ""},
        )
        self.assertTrue(resp.context["form"].errors.get("target_date"))


class ApiPredictTests(TestCase):
    def test_api_returns_json_prediction(self):
        resp = self.client.get(reverse("forecast:api_predict"), {"model": "sarimax", "date": "2026-03-15"})
        self.assertEqual(resp.status_code, 200)
        payload = json.loads(resp.content)
        self.assertEqual(payload["model_choice"], "sarimax")
        self.assertIn("predicted_temp_c", payload)

    def test_api_missing_date_is_400(self):
        resp = self.client.get(reverse("forecast:api_predict"), {"model": "sarimax"})
        self.assertEqual(resp.status_code, 400)

    def test_api_bad_date_format_is_400(self):
        resp = self.client.get(reverse("forecast:api_predict"), {"model": "sarimax", "date": "15-03-2026"})
        self.assertEqual(resp.status_code, 400)

    def test_api_unknown_model_is_400(self):
        resp = self.client.get(reverse("forecast:api_predict"), {"model": "prophet", "date": "2026-03-15"})
        self.assertEqual(resp.status_code, 400)

    def test_api_repeated_call_is_served_from_cache(self):
        params = {"model": "sarimax", "date": "2026-04-01"}
        first = self.client.get(reverse("forecast:api_predict"), params)
        second = self.client.get(reverse("forecast:api_predict"), params)
        self.assertFalse(json.loads(first.content)["from_cache"])
        self.assertTrue(json.loads(second.content)["from_cache"])


class EngineUnitTests(TestCase):
    def test_horizon_calculation(self):
        engine = ml.get_engine()
        import datetime
        horizon = engine.validate_target_date(datetime.date(2026, 1, 10))
        self.assertEqual(horizon, 10)
