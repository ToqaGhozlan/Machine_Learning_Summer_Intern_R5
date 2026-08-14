# Alexandria Temperature Forecaster

A Django web app that wraps a trained SARIMA + Fourier time series model. Enter tomorrow's
expected weather conditions and get a predicted temperature back instantly.

## What it does

- You fill in a form with 6 weather values (max/min temperature, precipitation, humidity,
  wind speed, solar radiation).
- The app validates your input (required, in-range, min ≤ max temperature).
- It loads the trained model (`models/sarima_fourier_model.pkl`) once and reuses it.
- It predicts the next day's temperature and shows it on the same page.

## Project structure

```
Task/
├── manage.py
├── db.sqlite3
├── requirements.txt
├── models/
│   └── sarima_fourier_model.pkl      # trained model from Task 5
├── weather_project/                  # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── weather/                          # Django app
    ├── forms.py                      # input form + validation
    ├── ml_model.py                   # loads model, runs prediction
    ├── views.py                      # connects form -> model -> template
    ├── urls.py
    └── templates/weather/
        ├── base.html
        └── index.html
```

## Requirements

- Python 3.10+
- pip

## Setup — run it locally in 5 steps

**1. Clone the repository and go into the project folder**
```bash
git clone <your-repo-url>
cd <project-folder>
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install the dependencies**
```bash
pip install -r requirements.txt
```

**4. Make sure the trained model is in place**

Confirm this file exists (it should already be in the repo):
```
models/sarima_fourier_model.pkl
```
If it's missing, the app will still run, but predictions will show a
"Model file not found" error instead of a result.

**5. Run the server**
```bash
python manage.py runserver
```
Then open **http://127.0.0.1:8000/** in your browser.

## Testing it yourself

- Fill in realistic values (e.g. max temp 28.5, min temp 18.0, humidity 65) and submit —
  you should see a predicted temperature.
- Try leaving a field empty — you'll get a "This field is required" message.
- Try typing letters into a number field — you'll get an invalid-value message.
- Try setting min temperature higher than max temperature — you'll get a validation error.
- Resize your browser to a phone width to check the layout still looks clean.

## Notes

- `DEBUG = True` in `weather_project/settings.py` is fine for local use only. If you deploy
  this publicly, set `DEBUG = False` and add your domain to `ALLOWED_HOSTS`.
- The prediction is always for the day right after the model's training data ends, since
  that's the one point in time the model can forecast without extra date input.
