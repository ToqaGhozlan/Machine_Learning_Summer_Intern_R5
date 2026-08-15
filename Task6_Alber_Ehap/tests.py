from django.test import TestCase

from django.urls import reverse


class WeatherFormTests(TestCase):


    def test_home_loads(self):

        response = self.client.get(
            reverse("home")
        )

        self.assertEqual(
            response.status_code,
            200
        )


        self.assertContains(
            response,
            "Weather Prediction"
        )



    def test_valid_prediction(self):

        response = self.client.post(

            reverse("home"),

            {

                "temperature": 28,

                "humidity": 55,

                "wind_speed": 18,

                "pressure": 1012,

            }

        )


        self.assertEqual(
            response.status_code,
            200
        )


        self.assertContains(
            response,
            "MODEL PREDICTION"
        )



    def test_invalid_humidity(self):

        response = self.client.post(

            reverse("home"),

            {

                "temperature": 28,

                "humidity": 120,

                "wind_speed": 18,

                "pressure": 1012,

            }

        )


        self.assertEqual(
            response.status_code,
            200
        )


        self.assertContains(
            response,
            "Ensure this value is less than or equal to 100"
        )