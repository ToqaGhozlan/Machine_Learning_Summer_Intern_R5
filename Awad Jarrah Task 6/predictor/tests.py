from django.test import TestCase

# Basic smoke tests. Run with: python manage.py test
# (Requires arma_manual.pkl / armax_auto.pkl / fourier_auto.pkl to be present
# in saved_models/, since PredictorConfig.ready() loads them on startup.)


class PredictPageTests(TestCase):
    def test_get_form_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Weather Temperature Predictor")

    def test_missing_fields_shows_validation_errors(self):
        response = self.client.post("/", {"model_choice": "armax_auto"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "This field is required for this model.")
