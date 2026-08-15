"""
Comprehensive test to verify Django predictions match the original Task 5 model
"""

import pickle
import json
import sys
from datetime import datetime

# Load the trained model
model_path = "outputs/Cairo/model/sarima_model.pkl"

print("="*70)
print("TASK 5 vs TASK 6 (DJANGO) PREDICTION VALIDATION")
print("="*70)

# Load the model
try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data.get('model')
        metadata = model_data.get('metadata', {})
    print(f"\n✓ Model loaded from: {model_path}")
    print(f"  Order: {metadata.get('order')}")
    print(f"  Seasonal Order: {metadata.get('seasonal_order')}")
    print(f"  Frequency: {metadata.get('freq')}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Test direct model predictions
print("\n" + "="*70)
print("DIRECT MODEL PREDICTIONS (Task 5 approach)")
print("="*70)

test_horizons = [1, 3, 7, 14, 28]
direct_predictions = {}

for horizon in test_horizons:
    try:
        forecast = model.get_forecast(steps=horizon)
        pred_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        
        # Get the final prediction at the horizon
        final_pred = float(pred_mean.iloc[-1])
        lower_ci = float(conf_int.iloc[-1, 0])
        upper_ci = float(conf_int.iloc[-1, 1])
        
        direct_predictions[horizon] = {
            'prediction': final_pred,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        }
        
        print(f"\n{horizon:2d} day(s) ahead: {final_pred:6.2f}°C")
        print(f"  95% CI: [{lower_ci:6.2f}, {upper_ci:6.2f}]°C")
    except Exception as e:
        print(f"\n✗ Error predicting {horizon} days ahead: {e}")

# Test Django predictions
print("\n" + "="*70)
print("DJANGO PREDICTIONS (Task 6)")
print("="*70)

import requests
import re

base_url = "http://localhost:8000"
session = requests.Session()

# Get CSRF token
response = session.get(f"{base_url}/")
csrf_match = re.search(r'csrfmiddlewaretoken["\']?\s*value=["\']([^"\']+)["\']', response.text)
csrf_token = csrf_match.group(1) if csrf_match else None

django_predictions = {}

for horizon in test_horizons:
    try:
        form_data = {
            'horizon_days': str(horizon),
            'csrfmiddlewaretoken': csrf_token
        }
        
        headers = {'Referer': f"{base_url}/"}
        
        response = session.post(f"{base_url}/", data=form_data, headers=headers)
        
        if response.status_code == 200:
            # Extract prediction from response
            pred_match = re.search(r'<span class="result-value">(\d+\.\d+)</span>', response.text)
            lower_match = re.search(r'Lower Bound:</span>\s*<span><strong>(\d+\.\d+)°C</strong>', response.text)
            upper_match = re.search(r'Upper Bound:</span>\s*<span><strong>(\d+\.\d+)°C</strong>', response.text)
            
            if pred_match:
                pred_value = float(pred_match.group(1))
                lower_ci = float(lower_match.group(1)) if lower_match else None
                upper_ci = float(upper_match.group(1)) if upper_match else None
                
                django_predictions[horizon] = {
                    'prediction': pred_value,
                    'lower_ci': lower_ci,
                    'upper_ci': upper_ci
                }
                
                print(f"\n{horizon:2d} day(s) ahead: {pred_value:6.2f}°C")
                if lower_ci and upper_ci:
                    print(f"  95% CI: [{lower_ci:6.2f}, {upper_ci:6.2f}]°C")
            else:
                print(f"\n✗ {horizon} days: Could not extract prediction from HTML")
        else:
            print(f"\n✗ {horizon} days: HTTP {response.status_code}")
    except Exception as e:
        print(f"\n✗ {horizon} days: {e}")

# Compare predictions
print("\n" + "="*70)
print("VALIDATION RESULTS")
print("="*70)

all_match = True
tolerance = 0.01  # Allow 0.01°C difference due to floating point

print(f"\nComparing predictions (tolerance: ±{tolerance}°C)")
print(f"\n{'Horizon':<12} {'Direct Model':<15} {'Django':<15} {'Difference':<15} {'Status':<10}")
print("-" * 70)

for horizon in test_horizons:
    if horizon in direct_predictions and horizon in django_predictions:
        direct = direct_predictions[horizon]['prediction']
        django = django_predictions[horizon]['prediction']
        diff = abs(direct - django)
        
        status = "✓ MATCH" if diff <= tolerance else "✗ MISMATCH"
        if diff > tolerance:
            all_match = False
        
        print(f"{horizon:2d} day(s)    {direct:6.2f}°C         {django:6.2f}°C         {diff:6.4f}°C         {status}")
    else:
        print(f"{horizon:2d} day(s)    MISSING")
        all_match = False

print("\n" + "="*70)
if all_match:
    print("✓ ALL PREDICTIONS MATCH - Django deployment is accurate!")
else:
    print("✗ PREDICTIONS DIFFER - Review model loading and inference logic")
print("="*70)

# Save test results
results = {
    'timestamp': datetime.now().isoformat(),
    'model_info': {
        'order': metadata.get('order'),
        'seasonal_order': metadata.get('seasonal_order'),
        'path': model_path
    },
    'direct_predictions': direct_predictions,
    'django_predictions': django_predictions,
    'validation_passed': all_match
}

with open('test_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nTest results saved to: test_results.json")
