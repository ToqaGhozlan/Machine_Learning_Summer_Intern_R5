const pickupDatetimeInput = document.getElementById('pickup_datetime');
if (pickupDatetimeInput.dataset.hasForm === 'false' && !pickupDatetimeInput.value) {
    const now = new Date();
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset()); 
    pickupDatetimeInput.value = now.toISOString().slice(0, 16);
}


const trafficHidden = document.getElementById('traffic_condition');
const trafficBtns = document.querySelectorAll('.traffic-btn');

function setTrafficActive(value) {
    trafficHidden.value = value;
    trafficBtns.forEach(btn => {
        if (btn.dataset.value === value) {
            btn.classList.add('border-secondary', 'bg-secondary-container/20', 'text-on-surface');
            btn.classList.remove('border-white/10', 'text-on-surface-variant');
        } else {
            btn.classList.remove('border-secondary', 'bg-secondary-container/20', 'text-on-surface');
            btn.classList.add('border-white/10', 'text-on-surface-variant');
        }
    });
}

trafficBtns.forEach(btn => {
    btn.addEventListener('click', () => setTrafficActive(btn.dataset.value));
});


if (trafficHidden.value) {
    setTrafficActive(trafficHidden.value);
}


const passengerInput = document.getElementById('passenger_count');
document.getElementById('passengerMinus').addEventListener('click', () => {
    passengerInput.value = Math.max(1, parseInt(passengerInput.value || 1) - 1);
});
document.getElementById('passengerPlus').addEventListener('click', () => {
    passengerInput.value = Math.min(6, parseInt(passengerInput.value || 1) + 1);
});


const nycBounds = [[40.5, -74.3], [41.8, -72.9]];

const map = L.map('map', {
    maxBounds: nycBounds,
    maxBoundsViscosity: 1.0,
    minZoom: 10
}).setView([40.7580, -73.9855], 12);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    bounds: nycBounds
}).addTo(map);

let pickupMarker = null;
let dropoffMarker = null;
let mode = 'pickup';

const pickupBtn = document.getElementById('pickupModeBtn');
const dropoffBtn = document.getElementById('dropoffModeBtn');
const pickupCoordEl = document.getElementById('pickup-coord');
const dropoffCoordEl = document.getElementById('dropoff-coord');
const distanceEl = document.getElementById('trip-distance');

const ACTIVE_CLASSES = ['bg-secondary-container', 'text-on-secondary-container', 'shadow-lg'];
const INACTIVE_CLASSES = ['text-on-surface-variant', 'hover:bg-white/5'];

function setMode(newMode) {
    mode = newMode;
    if (mode === 'pickup') {
        pickupBtn.classList.add(...ACTIVE_CLASSES);
        pickupBtn.classList.remove(...INACTIVE_CLASSES);
        dropoffBtn.classList.remove(...ACTIVE_CLASSES);
        dropoffBtn.classList.add(...INACTIVE_CLASSES);
    } else {
        dropoffBtn.classList.add(...ACTIVE_CLASSES);
        dropoffBtn.classList.remove(...INACTIVE_CLASSES);
        pickupBtn.classList.remove(...ACTIVE_CLASSES);
        pickupBtn.classList.add(...INACTIVE_CLASSES);
    }
}

pickupBtn.addEventListener('click', () => setMode('pickup'));
dropoffBtn.addEventListener('click', () => setMode('dropoff'));


function haversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const toRad = deg => deg * Math.PI / 180;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a = Math.sin(dLat / 2) ** 2 +
        Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(a));
}

function updateCoordsDisplay() {
    const pLat = document.getElementById('pickup_latitude').value;
    const pLon = document.getElementById('pickup_longitude').value;
    const dLat = document.getElementById('dropoff_latitude').value;
    const dLon = document.getElementById('dropoff_longitude').value;

    pickupCoordEl.textContent = (pLat && pLon) ? `${parseFloat(pLat).toFixed(4)}° N, ${Math.abs(parseFloat(pLon)).toFixed(4)}° W` : 'Not set';
    dropoffCoordEl.textContent = (dLat && dLon) ? `${parseFloat(dLat).toFixed(4)}° N, ${Math.abs(parseFloat(dLon)).toFixed(4)}° W` : 'Not set';

    if (pLat && pLon && dLat && dLon) {
        const km = haversineKm(parseFloat(pLat), parseFloat(pLon), parseFloat(dLat), parseFloat(dLon));
        distanceEl.textContent = `${km.toFixed(2)} km`;
    } else {
        distanceEl.textContent = '0.0 km';
    }
}

map.on('click', function (e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;

    if (mode === 'pickup') {
        if (pickupMarker) map.removeLayer(pickupMarker);
        pickupMarker = L.marker([lat, lon], {
            icon: L.divIcon({ className: '', html: '📍', iconSize: [24, 24] })
        }).addTo(map).bindPopup('Pickup').openPopup();

        document.getElementById('pickup_latitude').value = lat.toFixed(6);
        document.getElementById('pickup_longitude').value = lon.toFixed(6);

        setMode('dropoff');
    } else {
        if (dropoffMarker) map.removeLayer(dropoffMarker);
        dropoffMarker = L.marker([lat, lon], {
            icon: L.divIcon({ className: '', html: '🏁', iconSize: [24, 24] })
        }).addTo(map).bindPopup('Dropoff').openPopup();

        document.getElementById('dropoff_latitude').value = lat.toFixed(6);
        document.getElementById('dropoff_longitude').value = lon.toFixed(6);
    }

    updateCoordsDisplay();
});


document.getElementById('fareForm').addEventListener('submit', function (e) {
    const pLat = document.getElementById('pickup_latitude').value;
    const dLat = document.getElementById('dropoff_latitude').value;
    if (!pLat || !dLat) {
        e.preventDefault();
        alert('Please set both a pickup and a dropoff point on the map before submitting.');
    }
});


window.addEventListener('load', function () {
    const pLat = document.getElementById('pickup_latitude').value;
    const pLon = document.getElementById('pickup_longitude').value;
    const dLat = document.getElementById('dropoff_latitude').value;
    const dLon = document.getElementById('dropoff_longitude').value;

    if (pLat && pLon) {
        pickupMarker = L.marker([parseFloat(pLat), parseFloat(pLon)], {
            icon: L.divIcon({ className: '', html: '📍', iconSize: [24, 24] })
        }).addTo(map).bindPopup('Pickup');
    }
    if (dLat && dLon) {
        dropoffMarker = L.marker([parseFloat(dLat), parseFloat(dLon)], {
            icon: L.divIcon({ className: '', html: '🏁', iconSize: [24, 24] })
        }).addTo(map).bindPopup('Dropoff');
    }
    updateCoordsDisplay();
    map.invalidateSize();
});

