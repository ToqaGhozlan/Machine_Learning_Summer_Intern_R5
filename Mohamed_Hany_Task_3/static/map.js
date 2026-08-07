/**
 * map.js – Leaflet map controller + AJAX fare prediction
 */

"use strict";

const NYC_CENTER = [40.7580, -73.9855];
const INITIAL_ZOOM = 12;
const NYC_BOUNDS = { latMin: 40.45, latMax: 40.95, lonMin: -74.30, lonMax: -73.65 };

let pickupMarker  = null;
let dropoffMarker = null;
let routeLine     = null;
let clickState    = "pickup";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $pickupLat    = document.getElementById("pickup_latitude");
const $pickupLon    = document.getElementById("pickup_longitude");
const $dropoffLat   = document.getElementById("dropoff_latitude");
const $dropoffLon   = document.getElementById("dropoff_longitude");
const $pickupDisp   = document.getElementById("pickup-display");
const $dropoffDisp  = document.getElementById("dropoff-display");
const $pickupCard   = document.getElementById("pickup-card");
const $dropoffCard  = document.getElementById("dropoff-card");
const $hint         = document.getElementById("map-hint");
const $resetBtn     = document.getElementById("reset-btn");
const $predictBtn   = document.getElementById("predict-btn");
const $overlay      = document.getElementById("overlay");
const $errorBanner  = document.getElementById("error-banner");
const $errorText    = document.getElementById("error-text");
const $form         = document.getElementById("fare-form");
const $resultBox    = document.getElementById("result-box");
const $fareAmount   = document.getElementById("fare-amount");

const $dotPickup  = document.querySelector("#step-pickup  .step__dot");
const $dotDropoff = document.querySelector("#step-dropoff .step__dot");
const $dotPredict = document.querySelector("#step-predict .step__dot");

// ── Map ───────────────────────────────────────────────────────────────────────
const map = L.map("map", { center: NYC_CENTER, zoom: INITIAL_ZOOM, zoomControl: true });
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "© <a href='https://openstreetmap.org/copyright'>OpenStreetMap</a>",
}).addTo(map);

function makeIcon(type) {
  return L.divIcon({
    className: "",
    html: `<div class="map-marker map-marker--${type}"><span>${type === "pickup" ? "P" : "D"}</span></div>`,
    iconSize: [32, 32], iconAnchor: [16, 32],
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmt(lat, lon) { return `${lat.toFixed(5)}, ${lon.toFixed(5)}`; }

function inNYC(lat, lon) {
  return lat >= NYC_BOUNDS.latMin && lat <= NYC_BOUNDS.latMax &&
         lon >= NYC_BOUNDS.lonMin && lon <= NYC_BOUNDS.lonMax;
}

function showError(msg) { $errorBanner.hidden = false; $errorText.textContent = msg; }
function clearError()   { $errorBanner.hidden = true;  $errorText.textContent = ""; }

function setStep(step) {
  [$dotPickup, $dotDropoff, $dotPredict].forEach(d =>
    d.classList.remove("step__dot--active", "step__dot--done")
  );
  if (step === "pickup")  { $dotPickup.classList.add("step__dot--active"); }
  if (step === "dropoff") { $dotPickup.classList.add("step__dot--done"); $dotDropoff.classList.add("step__dot--active"); }
  if (step === "done")    { $dotPickup.classList.add("step__dot--done"); $dotDropoff.classList.add("step__dot--done"); $dotPredict.classList.add("step__dot--active"); }
}

function tryEnablePredict() {
  const ok = $pickupLat.value && $pickupLon.value && $dropoffLat.value && $dropoffLon.value;
  $predictBtn.disabled = !ok;
  if (ok) setStep("done");
}

function drawRoute() {
  if (!pickupMarker || !dropoffMarker) return;
  if (routeLine) map.removeLayer(routeLine);
  routeLine = L.polyline(
    [pickupMarker.getLatLng(), dropoffMarker.getLatLng()],
    { color: "#3b82f6", weight: 3, dashArray: "6 4", opacity: 0.85 }
  ).addTo(map);
}

function updatePickup(lat, lon) {
  $pickupLat.value = lat; $pickupLon.value = lon;
  $pickupDisp.textContent = fmt(lat, lon);
  $pickupCard.classList.add("is-set");
}
function updateDropoff(lat, lon) {
  $dropoffLat.value = lat; $dropoffLon.value = lon;
  $dropoffDisp.textContent = fmt(lat, lon);
  $dropoffCard.classList.add("is-set");
}

// ── Map clicks ────────────────────────────────────────────────────────────────
map.on("click", function (e) {
  const { lat, lng: lon } = e.latlng;
  if (!inNYC(lat, lon)) { showError("Click inside New York City boundaries."); return; }
  clearError();

  if (clickState === "pickup") {
    if (pickupMarker) map.removeLayer(pickupMarker);
    pickupMarker = L.marker([lat, lon], { icon: makeIcon("pickup"), draggable: true })
      .addTo(map).bindTooltip("Pickup", { permanent: true, direction: "top", offset: [0, -36] });
    pickupMarker.on("dragend", () => { const p = pickupMarker.getLatLng(); updatePickup(p.lat, p.lng); drawRoute(); tryEnablePredict(); });
    updatePickup(lat, lon);
    clickState = "dropoff"; setStep("dropoff");
    $hint.innerHTML = '🏁 Now click your <b>Dropoff</b> location';
    $resetBtn.hidden = false;

  } else if (clickState === "dropoff") {
    if (dropoffMarker) map.removeLayer(dropoffMarker);
    dropoffMarker = L.marker([lat, lon], { icon: makeIcon("dropoff"), draggable: true })
      .addTo(map).bindTooltip("Dropoff", { permanent: true, direction: "top", offset: [0, -36] });
    dropoffMarker.on("dragend", () => { const p = dropoffMarker.getLatLng(); updateDropoff(p.lat, p.lng); drawRoute(); tryEnablePredict(); });
    updateDropoff(lat, lon);
    drawRoute();
    map.fitBounds(L.latLngBounds(pickupMarker.getLatLng(), dropoffMarker.getLatLng()), { padding: [60, 60] });
    clickState = "done";
    $hint.innerHTML = '✅ Locations set! Fill the form and <b>Predict Fare</b>';
  }
  tryEnablePredict();
});

// ── Reset ─────────────────────────────────────────────────────────────────────
$resetBtn.addEventListener("click", function () {
  [pickupMarker, dropoffMarker, routeLine].forEach(l => { if (l) map.removeLayer(l); });
  pickupMarker = dropoffMarker = routeLine = null;
  clickState = "pickup";
  $pickupLat.value = $pickupLon.value = $dropoffLat.value = $dropoffLon.value = "";
  $pickupDisp.textContent = $dropoffDisp.textContent = "Click the map →";
  $pickupCard.classList.remove("is-set"); $dropoffCard.classList.remove("is-set");
  $predictBtn.disabled = true; $resetBtn.hidden = true;
  $hint.innerHTML = '📍 Click anywhere in NYC to set your <b>Pickup</b> location';
  clearError(); hideResult(); setStep("pickup");
  map.setView(NYC_CENTER, INITIAL_ZOOM);
});

// ── Result box ────────────────────────────────────────────────────────────────
function showResult(fare) {
  $fareAmount.textContent = "$" + fare.toFixed(2);
  $resultBox.hidden = false;
  $resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function hideResult() { $resultBox.hidden = true; }

// ── AJAX form submit ──────────────────────────────────────────────────────────
$form.addEventListener("submit", async function (e) {
  e.preventDefault();   // always intercept – no page navigation

  if (!$pickupLat.value || !$dropoffLat.value) {
    showError("Please set both pickup and dropoff on the map.");
    return;
  }

  const datetimeVal = document.getElementById("pickup_datetime").value;
  if (!datetimeVal) {
    showError("Please select a pickup date and time.");
    return;
  }
  const dateObj = new Date(datetimeVal);
  const year = dateObj.getFullYear();
  if (isNaN(year) || year < 2009 || year > 2015) {
    showError("pickup_datetime must be between years 2009 and 2015.");
    return;
  }

  clearError();
  hideResult();
  $overlay.hidden = false;

  const payload = {
    pickup_longitude:  parseFloat($pickupLon.value),
    pickup_latitude:   parseFloat($pickupLat.value),
    dropoff_longitude: parseFloat($dropoffLon.value),
    dropoff_latitude:  parseFloat($dropoffLat.value),
    passenger_count:   parseInt(document.getElementById("passenger_count").value, 10),
    car_condition:     document.getElementById("car_condition").value,
    weather:           document.getElementById("weather").value,
    traffic_condition: document.getElementById("traffic_condition").value,
    pickup_datetime:   document.getElementById("pickup_datetime").value,
  };

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await resp.json();
    $overlay.hidden = true;

    if (data.error) {
      showError(data.error);
    } else {
      showResult(data.fare);
      setStep("done");
      $dotPredict.classList.add("step__dot--done");
    }
  } catch (err) {
    $overlay.hidden = true;
    showError("Request failed: " + err.message);
  }
});

// ── Default datetime ──────────────────────────────────────────────────────────
(function () {
  // Set default date to a valid date within training period (2009–2015)
  document.getElementById("pickup_datetime").value = "2015-06-01T12:00";
})();

setStep("pickup");
