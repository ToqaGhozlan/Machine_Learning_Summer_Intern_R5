import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

pipeline = joblib.load('uber_fare_pipeline.pkl')

JFK_COORDS = (40.6413, -73.7781)


def haversine_distance(lat1, lon1, lat2, lon2):
    """بتحسب المسافة بالكيلومتر بين نقطتين على سطح الأرض (كروي)"""
    R = 6371  # نصف قطر الأرض بالكيلومتر
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """بتحسب زاوية الاتجاه (bearing) من نقطة لنقطة بالراديان"""
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_r - lon1_r
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    bearing = np.arctan2(x, y)
    return bearing  


def validate_input(form):
    """بترجع (errors_list, cleaned_data_dict). لو errors_list فاضية يبقى المدخلات سليمة"""
    errors = []
    data = {}

    # الحقول الرقمية المطلوبة، مع النطاق المنطقي لكل واحد (بناءً على بياناتنا الأصلية)
    numeric_fields = {
        'pickup_latitude': (-90, 90),
        'pickup_longitude': (-180, 180),
        'dropoff_latitude': (-90, 90),
        'dropoff_longitude': (-180, 180),
        'passenger_count': (1, 6),
    }

    for field, (min_val, max_val) in numeric_fields.items():
        raw = form.get(field, '').strip()
        if raw == '':
            errors.append(f"الحقل '{field}' مطلوب ومش ممكن يفضل فاضي.")
            continue
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"القيمة اللي دخلتها في '{field}' لازم تكون رقم.")
            continue
        if not (min_val <= value <= max_val):
            errors.append(f"القيمة في '{field}' لازم تكون بين {min_val} و {max_val}.")
            continue
        data[field] = value

    # التاريخ والوقت
    datetime_raw = form.get('trip_datetime', '').strip()
    if datetime_raw == '':
        errors.append("لازم تدخل تاريخ ووقت الرحلة.")
    else:
        try:
            data['trip_datetime'] = pd.to_datetime(datetime_raw)
        except Exception:
            errors.append("صيغة التاريخ/الوقت غير صحيحة.")

    # الحقول النصية (Dropdowns) - لازم تكون من القيم المسموحة بالظبط
    categorical_fields = {
        'car_condition': ['Bad', 'Good', 'Very Good', 'Excellent'],
        'traffic_condition': ['Flow Traffic', 'Dense Traffic', 'Congested Traffic'],
        'weather': ['sunny', 'rainy', 'windy', 'cloudy', 'stormy'],  # هنعدلها بعد ما تأكدلي القيم الفعلية
    }
    for field, allowed in categorical_fields.items():
        raw = form.get(field, '').strip()
        if raw == '':
            errors.append(f"الحقل '{field}' مطلوب.")
        elif raw not in allowed:
            errors.append(f"القيمة في '{field}' غير مسموحة.")
        else:
            data[field] = raw

    return errors, data


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    errors, data = validate_input(request.form)

    if errors:
        # لو فيه أي خطأ، نرجع لنفس الفورم مع رسالة الخطأ (مش صفحة نتيجة)
        return render_template('index.html', errors=errors)

    # استخراج الـ features الزمنية من التاريخ/الوقت
    dt = data['trip_datetime']
    day = dt.day
    month = dt.month
    weekday = dt.weekday()  # 0 = Monday
    year = dt.year
    hour = dt.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # حساب distance و bearing و jfk_dist من الإحداثيات
    distance = haversine_distance(
        data['pickup_latitude'], data['pickup_longitude'],
        data['dropoff_latitude'], data['dropoff_longitude']
    )
    bearing = calculate_bearing(
        data['pickup_latitude'], data['pickup_longitude'],
        data['dropoff_latitude'], data['dropoff_longitude']
    )
    jfk_dist = haversine_distance(
        data['dropoff_latitude'], data['dropoff_longitude'],
        JFK_COORDS[0], JFK_COORDS[1]
    )

    # تحويل الإحداثيات من degrees لـ radians (زي ما الموديل اتدرب بالظبط)
    pickup_lat_rad = np.radians(data['pickup_latitude'])
    pickup_lon_rad = np.radians(data['pickup_longitude'])
    dropoff_lat_rad = np.radians(data['dropoff_latitude'])
    dropoff_lon_rad = np.radians(data['dropoff_longitude'])

    # بناء DataFrame بنفس أسماء وترتيب أعمدة X_train بالظبط
    input_df = pd.DataFrame([{
        'Car Condition': data['car_condition'],
        'Weather': data['weather'],
        'Traffic Condition': data['traffic_condition'],
        'pickup_longitude': pickup_lon_rad,
        'pickup_latitude': pickup_lat_rad,
        'dropoff_longitude': dropoff_lon_rad,
        'dropoff_latitude': dropoff_lat_rad,
        'passenger_count': data['passenger_count'],
        'day': day,
        'month': month,
        'weekday': weekday,
        'year': year,
        'jfk_dist': jfk_dist,
        'distance': distance,
        'bearing': bearing,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
    }])

    # تنبؤ (الناتج بيكون log scale، لازم نرجعه بـ expm1)
    pred_log = pipeline.predict(input_df)
    predicted_fare = float(np.expm1(pred_log)[0])
    predicted_fare = round(predicted_fare, 2)

    return render_template('result.html', fare=predicted_fare, inputs=data)


if __name__ == '__main__':
    app.run(debug=True)