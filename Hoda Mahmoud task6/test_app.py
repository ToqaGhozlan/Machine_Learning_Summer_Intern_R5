import requests
import json
import re

# Test the prediction API
base_url = "http://localhost:8000"

# Create a session to maintain cookies
session = requests.Session()

# Get the prediction form
response = session.get(f"{base_url}/")
print(f"GET / - Status: {response.status_code}")
print(f"Page title in response: {'Temperature Prediction' in response.text}")
print(f"Model Status info present: {'Model Status' in response.text}")

# Test making a prediction via POST
print("\n" + "="*60)
print("Testing Prediction Request")
print("="*60)

# Get CSRF token from the form
csrf_match = re.search(r'csrfmiddlewaretoken["\']?\s*value=["\']([^"\']+)["\']', response.text)
if csrf_match:
    csrf_token = csrf_match.group(1)
    print(f"CSRF Token found: {csrf_token[:20]}...")
    
    # Also get from cookies
    csrf_cookie = session.cookies.get('csrftoken')
    if csrf_cookie:
        print(f"CSRF Cookie found: {csrf_cookie[:20]}...")
    
    # Submit form with 1 day ahead prediction
    form_data = {
        'horizon_days': '1',
        'csrfmiddlewaretoken': csrf_token
    }
    
    headers = {
        'Referer': f"{base_url}/"
    }
    
    response = session.post(f"{base_url}/", data=form_data, headers=headers)
    print(f"\nPOST / (1 day prediction) - Status: {response.status_code}")
    
    # Check if prediction result is in response
    if "Predicted Temperature" in response.text or "result" in response.text.lower():
        print("✓ Prediction result found in response")
        # Extract the prediction value
        temp_match = re.search(r'(\d+\.\d+)\s*°C', response.text)
        if temp_match:
            print(f"✓ Predicted temperature: {temp_match.group(1)}°C")
    elif "403" in response.text or response.status_code == 403:
        print("✗ CSRF error - check token handling")
    else:
        print("✗ Prediction result not found in response")
else:
    print("Could not find CSRF token in form")

# Test model info page
print("\n" + "="*60)
print("Testing Model Info Page")
print("="*60)

response = requests.get(f"{base_url}/info/")
print(f"GET /info/ - Status: {response.status_code}")
print(f"Model Information present: {'Model Information' in response.text}")
print(f"SARIMA config shown: {'SARIMA' in response.text and '2, 1, 2' in response.text}")

print("\n" + "="*60)
print("Testing Complete!")
print("="*60)
