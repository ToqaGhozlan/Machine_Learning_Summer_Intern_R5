# Weather Prediction System - Task 6 Deployment

A Django-based web application for weather temperature forecasting using SARIMA (Seasonal AutoRegressive Integrated Moving Average) time series modeling.

## Overview

This project deploys a trained SARIMA forecasting model through a professional Django web interface. The model predicts daily average temperatures for London, United Kingdom, based on historical weather patterns and seasonal cycles.

### Key Features

- **SARIMA Forecasting Model**: Trained on historical NASA POWER weather data for London
- **Django Web Interface**: Professional, responsive user interface
- **Real-time Predictions**: Get temperature forecasts for 1-28 days ahead
- **Confidence Intervals**: 95% confidence intervals around predictions
- **Model Information**: Detailed explanations of the model and methodology
- **Production-Ready**: Includes security best practices and environment configuration

## Project Structure

```
.
├── manage.py                          # Django management script
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
├── README.md                          # This file
│
├── weather_prediction/                # Django project configuration
│   ├── __init__.py
│   ├── settings.py                   # Django settings
│   ├── urls.py                       # URL routing
│   ├── wsgi.py                       # WSGI application
│   └── asgi.py                       # ASGI application
│
├── weather_app/                       # Weather prediction app
│   ├── __init__.py
│   ├── apps.py                       # App configuration
│   ├── models.py                     # Database models (empty for this app)
│   ├── views.py                      # View functions
│   ├── forms.py                      # Django forms
│   ├── urls.py                       # App URL routing
│   ├── ml_model.py                   # ML inference module
│   ├── admin.py                      # Django admin
│   ├── templates/
│   │   └── weather_app/
│   │       ├── base.html             # Base template
│   │       ├── prediction.html       # Prediction form/results
│   │       └── info.html             # Model information page
│   └── static/
│       └── css/                      # Custom CSS (optional)
│
├── outputs/                           # Task 5 outputs
│   └── Cairo/
│       ├── model/
│       │   └── sarima_model.pkl      # Trained SARIMA model
│       ├── cleaned/
│       ├── eda/
│       ├── figures/
│       └── ...
│
└── Performance.py                     # Task 5 training script
```

## Technologies Used

- **Django 6.1**: Web framework
- **Python 3.8+**: Programming language
- **Statsmodels**: SARIMA implementation
- **Pandas/NumPy**: Data processing
- **Bootstrap 5**: Frontend UI framework
- **SQLite**: Development database
- **Gunicorn** (optional): Production server

## Installation

### 1. Clone and Setup Virtual Environment

```bash
cd "d:\Cellalu internship\Hoda Mahmoud task5"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file based on `.env.example`:

```bash
copy .env.example .env
```

Edit `.env` and set your preferred values:
```
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

### 4. Django Setup

```bash
# Create database (SQLite)
python manage.py migrate

# Create superuser (optional, for admin panel)
python manage.py createsuperuser

# Collect static files (for production)
python manage.py collectstatic --noinput
```

## Running the Application

### Development Server

```bash
python manage.py runserver
```

The application will be available at:
```
http://localhost:8000
```

### Access Points

- **Prediction Form**: http://localhost:8000/
- **Model Information**: http://localhost:8000/info/
- **Django Admin**: http://localhost:8000/admin/ (if superuser created)

## Usage Guide

### Making Predictions

1. Navigate to the home page
2. Select your desired forecast horizon (1-28 days ahead)
3. Click "Generate Forecast"
4. View the predicted temperature with 95% confidence interval

### Understanding Results

The prediction shows:
- **Predicted Temperature**: The model's point forecast (°C)
- **Confidence Interval**: Range where the true value is likely to fall with 95% confidence
- **Model Info**: Technical details about the SARIMA model used

### Model Information Page

Visit `/info/` to learn:
- SARIMA model configuration
- Training data statistics
- How the model works
- Model limitations and disclaimers

## Model Details

### SARIMA Configuration

- **Order**: (2, 1, 2)
  - p=2: AutoRegressive components
  - d=1: First-order differencing
  - q=2: Moving Average components

- **Seasonal Order**: (0, 0, 0, 7)
  - 7-day seasonal period (weekly patterns)
  - No seasonal differencing applied

### Training Data

- **Location**: London, United Kingdom (51.51°N, 0.13°W)
- **Source**: NASA POWER climate dataset
- **Time Range**: 2022-01-01 to 2022-12-31
- **Variable**: Daily average temperature (T2M)
- **Training Set**: 2022-01-01 to 2022-12-03
- **Test Set**: 2022-12-04 to 2022-12-31

### Model Performance

| Metric | SARIMA | Persistence Baseline |
|--------|--------|----------------------|
| MAE | 3.11°C | 3.40°C |
| RMSE | 3.44°C | 3.73°C |
| SMAPE | 67.73% | 87.14% |

The model outperforms the simple persistence baseline, validating the utility of the learned patterns.

## Testing

### Verify Installation

```bash
# Check model is loaded
python manage.py shell
>>> from weather_app.ml_model import get_model_info
>>> info = get_model_info()
>>> print(info)
```

### Test Predictions Programmatically

```python
from weather_app.ml_model import predict_temperature

# Single prediction
result = predict_temperature(horizon_days=1)
print(f"Tomorrow's temperature: {result['prediction']:.2f}°C")
print(f"Confidence interval: [{result['lower_ci']:.2f}, {result['upper_ci']:.2f}]°C")
```

### Validation Against Task 5

The Django predictions should match the original Task 5 SARIMA model within floating-point tolerance:

```bash
# Run original model
python Performance.py

# Compare output from Django with Task 5 forecasting_summary.md
```

## Deployment (Production)

### Environment Variables

Update `.env` for production:

```
DEBUG=False
SECRET_KEY=your-secure-secret-key-here
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
```

### Security Settings

The `settings.py` includes production security configurations:
- SSL/HTTPS enforcement
- Secure cookies
- CSP headers
- XSS protection

Enable them by setting `DEBUG=False`.

### Deploy to Render.com

1. Create `Procfile`:
```
web: gunicorn weather_prediction.wsgi
```

2. Install production requirements:
```bash
pip install gunicorn
```

3. Update `requirements.txt`:
```bash
pip freeze > requirements.txt
```

4. Push to GitHub and connect to Render

### Deploy to Railway.app

1. Connect repository to Railway
2. Set environment variables in Railway dashboard
3. Railway automatically detects Django and deploys

## Troubleshooting

### Model Not Found

**Error**: `Model file not found at outputs/Cairo/model/sarima_model.pkl`

**Solution**:
```bash
python Performance.py
```
This regenerates the model file.

### Import Errors

**Error**: `No module named 'statsmodels'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Database Errors

**Error**: `ProgrammingError` or `OperationalError`

**Solution**:
```bash
python manage.py migrate
```

### Static Files Missing

**Error**: CSS/images not loading

**Solution**:
```bash
python manage.py collectstatic --noinput
```

## Screenshots

### Prediction Form
[Screenshot of the prediction interface would go here]

### Prediction Result
[Screenshot of the result with confidence interval would go here]

### Model Information
[Screenshot of the info page would go here]

## Important Notes

1. **Data Requirement**: This model requires the trained SARIMA model pickle file at `outputs/Cairo/model/sarima_model.pkl`

2. **Historical Context**: The model was trained on 2022 London weather data. Performance may vary for predictions far into the future.

3. **Confidence Intervals**: The 95% confidence intervals assume historical patterns continue. Confidence widens as forecast horizon increases.

4. **No Retraining**: The current implementation does not retrain the model on each prediction. To retrain, run `Performance.py` manually.

5. **Static Files**: In production, use a web server (Nginx) or CDN for serving static files and media.

## Requirements Details

Key Python packages:
- `Django>=4.2`: Web framework
- `pandas>=1.3`: Data manipulation
- `numpy>=1.21`: Numerical computing
- `statsmodels>=0.13`: SARIMA models
- `joblib>=1.0`: Model serialization
- `python-decouple>=3.8`: Environment configuration

See `requirements.txt` for complete list.

## Contributing

To modify or extend this project:

1. Modify `Performance.py` to retrain the model
2. Update form fields in `weather_app/forms.py`
3. Modify prediction logic in `weather_app/ml_model.py`
4. Update templates in `weather_app/templates/`
5. Restart Django server: `python manage.py runserver`

## License

This is an educational project for the Cellalu internship program.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review Django documentation: https://docs.djangoproject.com/
3. Review statsmodels documentation: https://www.statsmodels.org/

---

**Task 6 Completion Checklist**:
- [x] Django project created and configured
- [x] Weather prediction app implemented
- [x] Task 5 SARIMA model integrated
- [x] Model serialized and loaded on startup
- [x] Prediction form with validation
- [x] Professional UI with Bootstrap 5
- [x] Responsive, mobile-friendly design
- [x] Confidence intervals displayed
- [x] Static files and configuration set up
- [x] Requirements.txt updated
- [x] README completed
- [x] .gitignore updated
- [x] Local development server working
- [x] Optional deployment configuration included

**Last Updated**: 2024
