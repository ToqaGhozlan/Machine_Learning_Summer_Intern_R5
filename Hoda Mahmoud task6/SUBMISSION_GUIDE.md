# Task 6 Submission Guide - Running and Testing the Application

## Quick Start

### 1. Install Dependencies
```bash
cd "d:\Cellalu internship\Hoda Mahmoud task5"
pip install -r requirements.txt
```

### 2. Setup Database
```bash
python manage.py migrate
```

### 3. Run Development Server
```bash
python manage.py runserver
```

The server will start at: **http://localhost:8000**

### 4. Access the Application
- **Prediction Form**: http://localhost:8000/
- **Model Information**: http://localhost:8000/info/

## How to Use the Application

### Making a Prediction

1. **Open the home page** (http://localhost:8000/)
2. **Select a forecast horizon** - Choose from 1-28 days:
   - "1 day ahead"
   - "2 days ahead"
   - "3 days ahead"
   - "4 days ahead"
   - "5 days ahead"
   - "1 week ahead" (7 days)
   - "2 weeks ahead" (14 days)
   - "3 weeks ahead" (21 days)
   - "4 weeks ahead" (28 days)
3. **Click "Generate Forecast"** button
4. **View the result** on the same page:
   - Predicted temperature in °C
   - 95% confidence interval (lower and upper bounds)
   - Model information

### Viewing Model Information

1. **Click "Model Info"** link in navigation
2. **Read about**:
   - SARIMA model configuration
   - Training data statistics
   - How predictions work
   - Model limitations

## Screenshots for Submission

You need to submit **3 screenshots**:

### Screenshot 1: Prediction Form (Empty)
1. Open http://localhost:8000/
2. **Do NOT submit the form yet**
3. Take screenshot showing:
   - Navigation bar with "Predict" and "Model Info" links
   - "Temperature Forecast" heading
   - "Model Status: Ready to predict" indicator
   - All forecast horizon radio buttons (1 day through 4 weeks)
   - "Generate Forecast" button
   - "About This Model" information card at bottom

**File name**: `screenshot_01_prediction_form.png` or `.jpg`

### Screenshot 2: Prediction Result
1. On the home page (http://localhost:8000/)
2. Select "1 day ahead" (or any horizon you prefer)
3. Click "Generate Forecast"
4. Take screenshot showing:
   - The prediction form (still visible above)
   - Green result card with:
     - "Predicted Temperature" label
     - Large number (e.g., "5.00°C")
     - "Forecast Horizon: 1 day" information
     - 95% Confidence Interval showing [lower, upper]°C
   - Model info below result

**File name**: `screenshot_02_prediction_result.png` or `.jpg`

### Screenshot 3: Model Information Page
1. Click "Model Info" link in the navigation
2. Scroll to see the full page
3. Take screenshot showing:
   - "Model Information" heading
   - "Status: Model Loaded and Ready"
   - Model Parameters table (Order, Seasonal Order, Seasonal Period, Frequency)
   - Data Statistics table (Mean, Std Dev, Min, Max temperatures)
   - Training Information (Training End Date, Test Dates)
   - About SARIMA section explaining the model
   - How Predictions Work section
   - Model Performance Metrics table
   - Limitations section

**File name**: `screenshot_03_model_info_page.png` or `.jpg`

**Alternative**: If page is long, you can take 2 screenshots:
- `screenshot_03a_model_info_top.png` (Model parameters and statistics)
- `screenshot_03b_model_info_bottom.png` (SARIMA explanation and limitations)

## Testing the Application Locally

### Test 1: Basic Functionality
```bash
python test_app.py
```

Expected output:
```
GET / - Status: 200
Page title in response: True
Model Status info present: True

============================================================
Testing Prediction Request
============================================================
CSRF Token found: ...
POST / (1 day prediction) - Status: 200
✓ Prediction result found in response
✓ Predicted temperature: X.XXX°C

GET /info/ - Status: 200
Model Information present: True
SARIMA config shown: True
```

### Test 2: Prediction Validation
```bash
python validate_predictions.py
```

Expected output includes:
```
✓ Model loaded from: outputs/Cairo/model/sarima_model.pkl
  Order: (2, 1, 2)
  Seasonal Order: (0, 0, 0, 7)

... (predictions for different horizons) ...

✓ ALL PREDICTIONS MATCH - Django deployment is accurate!
```

## Troubleshooting

### Issue: "Model file not found"
**Solution**: Run Performance.py to regenerate the model:
```bash
python Performance.py
```

### Issue: "Port 8000 already in use"
**Solution**: Use a different port:
```bash
python manage.py runserver 8001
```

### Issue: Form submission gives 403 Forbidden
**Solution**: This is normal on first POST if cookies weren't set. Try:
1. Refresh the page (Ctrl+R)
2. Submit the form again

### Issue: CSS/Bootstrap styling not showing
**Solution**: This is expected in development (using CDN):
1. Check your internet connection
2. Bootstrap loads from CDN: `cdn.jsdelivr.net`
3. Styling will work fine when deployed

## File Structure Quick Reference

```
d:\Cellalu internship\Hoda Mahmoud task5\
├── manage.py                           # Django management script
├── requirements.txt                    # Python dependencies
├── README.md                           # Full documentation
├── TASK6_IMPLEMENTATION_SUMMARY.md    # This file
├── .env.example                        # Environment template
├── Performance.py                      # Task 5 script (modified to save model)
│
├── weather_prediction/                 # Django project config
│   ├── settings.py                    # Configuration
│   ├── urls.py                        # Main URL routing
│   ├── wsgi.py                        # WSGI entry point
│   └── asgi.py                        # ASGI entry point
│
├── weather_app/                        # Weather prediction app
│   ├── views.py                       # Prediction & info views
│   ├── forms.py                       # Forecast horizon form
│   ├── urls.py                        # App URL routing
│   ├── ml_model.py                    # Model loading & prediction
│   └── templates/weather_app/
│       ├── base.html                 # Base template
│       ├── prediction.html           # Main prediction page
│       └── info.html                 # Model info page
│
├── outputs/Cairo/model/
│   └── sarima_model.pkl               # Trained SARIMA model
│
└── tests/
    ├── test_app.py                   # Basic tests
    └── validate_predictions.py       # Validation tests
```

## Performance Metrics

The deployed SARIMA model achieves:
- **MAE**: 3.11°C (Mean Absolute Error)
- **RMSE**: 3.44°C (Root Mean Squared Error)
- **Performance**: Better than persistence baseline

Test predictions are accurate to within ±0.005°C of the original Task 5 model.

## Before Submitting

**Checklist**:
- [ ] Django server runs without errors: `python manage.py runserver`
- [ ] Home page loads: http://localhost:8000/
- [ ] Can select forecast horizon
- [ ] Can submit prediction form
- [ ] Prediction displays with result
- [ ] Confidence intervals shown
- [ ] Model Info page loads: http://localhost:8000/info/
- [ ] Test script passes: `python test_app.py`
- [ ] Validation passes: `python validate_predictions.py`
- [ ] 3 screenshots taken and saved
- [ ] README is comprehensive
- [ ] All files committed to Git

## Submission Files

You should submit:
1. **This entire project directory** with all files
2. **3 screenshots** (or 4 if splitting model info page):
   - `screenshot_01_prediction_form.png`
   - `screenshot_02_prediction_result.png`
   - `screenshot_03_model_info_page.png`
3. **README.md** (already included in project)
4. **TASK6_IMPLEMENTATION_SUMMARY.md** (detailed implementation notes)
5. **test_results.json** (validation test results - auto-generated)

## Important URLs

- **Home/Prediction**: http://localhost:8000/
- **Model Info**: http://localhost:8000/info/
- **Admin Panel** (if needed): http://localhost:8000/admin/

## Running for Submission Demo

```bash
# 1. Navigate to project
cd "d:\Cellalu internship\Hoda Mahmoud task5"

# 2. Ensure virtual environment is activated (if using one)
# .venv\Scripts\activate

# 3. Start the server
python manage.py runserver

# 4. Open browser to http://localhost:8000
# 5. Test the application as described above
# 6. Take screenshots
# 7. Stop server (Ctrl+C in terminal)
```

## Key Implementation Details

### Model Integration
- SARIMA model loaded at Django startup
- Model pickled with metadata from Task 5
- No retraining on each prediction
- Predictions match Task 5 output exactly

### Input Validation
- Forecast horizon: 1-28 days (validated)
- Radio button selection (no invalid inputs possible)
- CSRF protection on all POST requests
- User-friendly error messages

### UI/UX Features
- Responsive Bootstrap 5 design
- Professional color scheme (purple gradient)
- Mobile-friendly layout
- Clear forecast result display
- Confidence intervals explained
- Model information accessible

### Code Quality
- Clean separation of concerns (views, forms, ml_model)
- Comprehensive docstrings
- Error handling with logging
- Security best practices
- Production-ready configuration

---

**Ready to Submit!** ✓

If you have any issues running the application, check the README.md or TASK6_IMPLEMENTATION_SUMMARY.md for detailed troubleshooting.
