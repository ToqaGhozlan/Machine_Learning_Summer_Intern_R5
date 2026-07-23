document.getElementById('trip-date').value = new Date().toISOString().slice(0, 10);

const NYC_CENTER = [40.7128, -74.0060];
const map = L.map('map').setView(NYC_CENTER, 12);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
  maxZoom: 19
}).addTo(map);

let pickupMarker = null;
let dropoffMarker = null;
let routeLine = null;

const pickupIcon = L.divIcon({
  className: '',
  html: '<div style="width:16px;height:16px;background:#34c759;border-radius:50%;border:2px solid white;"></div>',
  iconSize: [16, 16]
});
const dropoffIcon = L.divIcon({
  className: '',
  html: '<div style="width:16px;height:16px;background:#ff453a;border-radius:50%;border:2px solid white;"></div>',
  iconSize: [16, 16]
});

function currentMode() {
  return document.querySelector('input[name="click-mode"]:checked').value;
}

function updateModeLabel() {
  const mode = currentMode();
  document.getElementById('mode-label').textContent = mode === 'pickup' ? 'Pickup' : 'Drop-off';
}
document.querySelectorAll('input[name="click-mode"]').forEach(el =>
  el.addEventListener('change', updateModeLabel)
);

function updateCoordBoxes() {
  document.getElementById('pickup-coord').textContent = pickupMarker
    ? `(${pickupMarker.getLatLng().lat.toFixed(5)}, ${pickupMarker.getLatLng().lng.toFixed(5)})`
    : 'not set';
  document.getElementById('dropoff-coord').textContent = dropoffMarker
    ? `(${dropoffMarker.getLatLng().lat.toFixed(5)}, ${dropoffMarker.getLatLng().lng.toFixed(5)})`
    : 'not set';
}

function updateRoute() {
  if (routeLine) { map.removeLayer(routeLine); routeLine = null; }
  if (pickupMarker && dropoffMarker) {
    routeLine = L.polyline([pickupMarker.getLatLng(), dropoffMarker.getLatLng()], {
      color: '#4da3ff', weight: 3, dashArray: '6 6'
    }).addTo(map);
  }
}

map.on('click', (e) => {
  const mode = currentMode();
  if (mode === 'pickup') {
    if (pickupMarker) { pickupMarker.setLatLng(e.latlng); }
    else { pickupMarker = L.marker(e.latlng, { icon: pickupIcon }).addTo(map); }
    // auto-advance to drop-off mode, like the reference UI
    document.getElementById('mode-dropoff').checked = true;
  } else {
    if (dropoffMarker) { dropoffMarker.setLatLng(e.latlng); }
    else { dropoffMarker = L.marker(e.latlng, { icon: dropoffIcon }).addTo(map); }
  }
  updateModeLabel();
  updateCoordBoxes();
  updateRoute();
});

document.getElementById('clear-btn').addEventListener('click', () => {
  if (pickupMarker) { map.removeLayer(pickupMarker); pickupMarker = null; }
  if (dropoffMarker) { map.removeLayer(dropoffMarker); dropoffMarker = null; }
  if (routeLine) { map.removeLayer(routeLine); routeLine = null; }
  document.getElementById('mode-pickup').checked = true;
  updateModeLabel();
  updateCoordBoxes();
  hideResults();
});

document.getElementById('passenger-count').addEventListener('input', (e) => {
  document.getElementById('passenger-value').textContent = e.target.value;
});

function showError(msg) {
  const box = document.getElementById('error-box');
  box.textContent = msg;
  box.style.display = 'block';
}
function hideError() {
  document.getElementById('error-box').style.display = 'none';
}
function hideResults() {
  document.getElementById('results-content').style.display = 'none';
  document.getElementById('placeholder-text').style.display = 'block';
}

document.getElementById('predict-btn').addEventListener('click', async () => {
  hideError();

  if (!pickupMarker || !dropoffMarker) {
    showError('Please click the map to set both a pickup and a drop-off point.');
    return;
  }

  const payload = {
    pickup_lat: pickupMarker.getLatLng().lat,
    pickup_lon: pickupMarker.getLatLng().lng,
    dropoff_lat: dropoffMarker.getLatLng().lat,
    dropoff_lon: dropoffMarker.getLatLng().lng,
    passenger_count: document.getElementById('passenger-count').value,
    date: document.getElementById('trip-date').value,
    time: document.getElementById('trip-time').value,
  };

  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.textContent = 'Predicting...';

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await resp.json();

    if (!resp.ok) {
      showError(data.error || 'Something went wrong while predicting the fare.');
      return;
    }

    document.getElementById('out-fare').textContent = `$${data.predicted_fare.toFixed(2)}`;
    document.getElementById('out-distance').textContent = `${data.trip_distance_km.toFixed(2)} km`;
    document.getElementById('out-bearing').textContent = `${data.direction_deg}\u00b0`;

    document.getElementById('info-day').textContent = data.day;
    document.getElementById('info-month').textContent = data.month_year;
    document.getElementById('info-hour').textContent = data.hour;
    document.getElementById('info-rush').textContent = data.is_rush_hour ? 'Yes' : 'No';
    document.getElementById('info-weekend').textContent = data.is_weekend ? 'Yes' : 'No';
    document.getElementById('info-pickup-center').textContent = `${data.pickup_to_center_km.toFixed(2)} km`;
    document.getElementById('info-dropoff-center').textContent = `${data.dropoff_to_center_km.toFixed(2)} km`;

    document.getElementById('results-content').style.display = 'block';
    document.getElementById('placeholder-text').style.display = 'none';
  } catch (err) {
    showError('Could not reach the server. Please try again.');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Predict fare';
  }
});
