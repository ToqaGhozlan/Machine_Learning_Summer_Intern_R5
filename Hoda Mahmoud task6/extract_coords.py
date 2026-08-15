"""
Extract coordinates from NASA POWER API response
"""
import json

json_path = 'outputs/Cairo/raw/power_response.json'
try:
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    geometry = data.get('geometry', {})
    coordinates = geometry.get('coordinates', [])
    properties = data.get('properties', {})
    
    print('=' * 80)
    print('ACTUAL NASA POWER API RESPONSE COORDINATES')
    print('=' * 80)
    print()
    print(f'Geometry coordinates: {coordinates}')
    print(f'  Format: [longitude, latitude]')
    if coordinates:
        lon, lat = coordinates[0], coordinates[1]
        print(f'  Longitude: {lon}')
        print(f'  Latitude: {lat}')
    print()
    
    geom_info = properties.get('geomInfo', {})
    print('GeomInfo from properties:')
    print(json.dumps(geom_info, indent=2))
    print()
    
    # Check date range
    param_data = properties.get('parameter', {})
    if 'T2M' in param_data:
        dates = sorted(param_data['T2M'].keys())
        print(f'Data range: {dates[0]} to {dates[-1]}')
        print(f'Total days: {len(dates)}')
        
except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc()
