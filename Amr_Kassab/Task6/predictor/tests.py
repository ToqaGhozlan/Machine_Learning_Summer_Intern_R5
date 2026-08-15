"""
Test suite for the Weather Prediction Django application.
Validates model integration, form validation, views, and API endpoint.
"""

import json
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_project.settings')
django.setup()

from django.test import TestCase, Client
from predictor.forms import WeatherPredictionForm
from predictor.ml_service import predict_temperature, get_model_metrics, get_feature_ranges


class ModelIntegrationTests(TestCase):
    """Verify that ML models load and produce valid predictions."""

    def test_gradient_boosting_prediction(self):
        result = predict_temperature({
            'relative_humidity': 65,
            'precipitation': 0.0,
            'wind_speed': 12.5,
            'cloud_cover': 30,
            'surface_pressure': 1013.25,
            'hour': 14,
            'month': 7,
        }, model_name='gradient_boosting')
        self.assertIn('temperature', result)
        self.assertIsInstance(result['temperature'], float)
        self.assertGreater(result['temperature'], -10)
        self.assertLess(result['temperature'], 55)

    def test_random_forest_prediction(self):
        result = predict_temperature({
            'relative_humidity': 45,
            'precipitation': 0.0,
            'wind_speed': 8,
            'cloud_cover': 80,
            'surface_pressure': 1010,
            'hour': 3,
            'month': 1,
        }, model_name='random_forest')
        self.assertIn('temperature', result)
        self.assertIsInstance(result['temperature'], float)

    def test_confidence_interval_present(self):
        result = predict_temperature({
            'relative_humidity': 50,
            'precipitation': 0,
            'wind_speed': 10,
            'cloud_cover': 50,
            'surface_pressure': 1013,
            'hour': 12,
            'month': 6,
        })
        self.assertIn('confidence_low', result)
        self.assertIn('confidence_high', result)
        self.assertLess(result['confidence_low'], result['temperature'])
        self.assertGreater(result['confidence_high'], result['temperature'])

    def test_model_metrics_loadable(self):
        metrics = get_model_metrics()
        self.assertIn('gradient_boosting', metrics)
        self.assertIn('random_forest', metrics)
        self.assertIn('MAE', metrics['gradient_boosting'])

    def test_feature_ranges_loadable(self):
        ranges = get_feature_ranges()
        self.assertIn('temperature', ranges)
        self.assertIn('relative_humidity', ranges)

    def test_summer_vs_winter_temperature_difference(self):
        summer = predict_temperature({
            'relative_humidity': 20, 'precipitation': 0, 'wind_speed': 5,
            'cloud_cover': 0, 'surface_pressure': 1005, 'hour': 14, 'month': 8,
        })
        winter = predict_temperature({
            'relative_humidity': 80, 'precipitation': 5, 'wind_speed': 20,
            'cloud_cover': 90, 'surface_pressure': 1020, 'hour': 3, 'month': 1,
        })
        self.assertGreater(summer['temperature'], winter['temperature'],
                           "Summer prediction should be higher than winter prediction")


class FormValidationTests(TestCase):
    """Verify form validation handles edge cases properly."""

    def test_valid_form(self):
        form = WeatherPredictionForm(data={
            'relative_humidity': 65,
            'precipitation': 0,
            'wind_speed': 12.5,
            'cloud_cover': 30,
            'surface_pressure': 1013.25,
            'hour': 14,
            'month': 7,
            'model_choice': 'gradient_boosting',
        })
        self.assertTrue(form.is_valid(), f"Form errors: {form.errors}")

    def test_empty_form_invalid(self):
        form = WeatherPredictionForm(data={})
        self.assertFalse(form.is_valid())

    def test_out_of_range_humidity(self):
        form = WeatherPredictionForm(data={
            'relative_humidity': 150,
            'precipitation': 0,
            'wind_speed': 12.5,
            'cloud_cover': 30,
            'surface_pressure': 1013.25,
            'hour': 14,
            'month': 7,
        })
        self.assertFalse(form.is_valid())
        self.assertIn('relative_humidity', form.errors)

    def test_negative_wind_speed(self):
        form = WeatherPredictionForm(data={
            'relative_humidity': 65,
            'precipitation': 0,
            'wind_speed': -5,
            'cloud_cover': 30,
            'surface_pressure': 1013.25,
            'hour': 14,
            'month': 7,
        })
        self.assertFalse(form.is_valid())
        self.assertIn('wind_speed', form.errors)

    def test_pressure_out_of_range(self):
        form = WeatherPredictionForm(data={
            'relative_humidity': 65,
            'precipitation': 0,
            'wind_speed': 12.5,
            'cloud_cover': 30,
            'surface_pressure': 500,
            'hour': 14,
            'month': 7,
        })
        self.assertFalse(form.is_valid())
        self.assertIn('surface_pressure', form.errors)


class ViewTests(TestCase):
    """Verify Django views respond correctly."""

    def setUp(self):
        self.client = Client()

    def test_homepage_get(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Cairo Weather Predictor')
        self.assertContains(response, 'Relative Humidity')

    def test_homepage_post_valid(self):
        response = self.client.post('/', {
            'relative_humidity': 65,
            'precipitation': 0,
            'wind_speed': 12.5,
            'cloud_cover': 30,
            'surface_pressure': 1013.25,
            'hour': 14,
            'month': 7,
            'model_choice': 'gradient_boosting',
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Prediction Result')
        self.assertContains(response, '°C')

    def test_homepage_post_invalid(self):
        response = self.client.post('/', {
            'relative_humidity': '',
            'precipitation': 0,
            'wind_speed': 12.5,
            'cloud_cover': 30,
            'surface_pressure': 1013.25,
            'hour': 14,
            'month': 7,
        })
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'Prediction Result')


class APITests(TestCase):
    """Verify the JSON API endpoint."""

    def setUp(self):
        self.client = Client()

    def test_api_valid_request(self):
        response = self.client.post(
            '/api/predict/',
            data=json.dumps({
                'relative_humidity': 65,
                'precipitation': 0,
                'wind_speed': 12.5,
                'cloud_cover': 30,
                'surface_pressure': 1013.25,
                'hour': 14,
                'month': 7,
            }),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('temperature', data)
        self.assertIn('confidence_low', data)

    def test_api_missing_fields(self):
        response = self.client.post(
            '/api/predict/',
            data=json.dumps({'relative_humidity': 65}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)

    def test_api_get_not_allowed(self):
        response = self.client.get('/api/predict/')
        self.assertEqual(response.status_code, 405)

    def test_api_invalid_json(self):
        response = self.client.post(
            '/api/predict/',
            data='not json',
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)

    def test_api_with_model_selection(self):
        response = self.client.post(
            '/api/predict/',
            data=json.dumps({
                'relative_humidity': 45,
                'precipitation': 0,
                'wind_speed': 8,
                'cloud_cover': 80,
                'surface_pressure': 1010,
                'hour': 3,
                'month': 1,
                'model': 'random_forest',
            }),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['model_used'], 'random_forest')
