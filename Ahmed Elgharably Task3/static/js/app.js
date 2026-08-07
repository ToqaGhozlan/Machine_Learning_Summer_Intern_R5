// Uber Fare Predictor — map interaction + AJAX submit
(function () {
  const NYC_CENTER = [40.7484, -73.9857];
  const map = L.map("map").setView(NYC_CENTER, 12);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap contributors",
    maxZoom: 18,
  }).addTo(map);

  let pickupMarker = null;
  let dropoffMarker = null;
  let routeLine = null;

  const pickupIcon = L.divIcon({ className: "map-marker pickup-marker", html: "P", iconSize: [26, 26] });
  const dropoffIcon = L.divIcon({ className: "map-marker dropoff-marker", html: "D", iconSize: [26, 26] });

  function setField(id, value) {
    document.getElementById(id).value = value.toFixed(6);
  }

  function drawRoute() {
    if (routeLine) map.removeLayer(routeLine);
    if (pickupMarker && dropoffMarker) {
      routeLine = L.polyline([pickupMarker.getLatLng(), dropoffMarker.getLatLng()], {
        color: "#1c7293",
        weight: 3,
        dashArray: "6 6",
      }).addTo(map);
    }
  }

  map.on("click", function (e) {
    const { lat, lng } = e.latlng;
    if (!pickupMarker) {
      pickupMarker = L.marker(e.latlng, { icon: pickupIcon, draggable: true }).addTo(map);
      setField("pickup_lat", lat);
      setField("pickup_lon", lng);
      pickupMarker.on("drag", (ev) => {
        setField("pickup_lat", ev.target.getLatLng().lat);
        setField("pickup_lon", ev.target.getLatLng().lng);
        drawRoute();
      });
    } else if (!dropoffMarker) {
      dropoffMarker = L.marker(e.latlng, { icon: dropoffIcon, draggable: true }).addTo(map);
      setField("dropoff_lat", lat);
      setField("dropoff_lon", lng);
      dropoffMarker.on("drag", (ev) => {
        setField("dropoff_lat", ev.target.getLatLng().lat);
        setField("dropoff_lon", ev.target.getLatLng().lng);
        drawRoute();
      });
      drawRoute();
    }
    // Third click onward: do nothing until reset, so accidental extra clicks don't move points.
  });

  document.getElementById("resetPoints").addEventListener("click", function () {
    if (pickupMarker) { map.removeLayer(pickupMarker); pickupMarker = null; }
    if (dropoffMarker) { map.removeLayer(dropoffMarker); dropoffMarker = null; }
    if (routeLine) { map.removeLayer(routeLine); routeLine = null; }
    ["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"].forEach((id) => {
      document.getElementById(id).value = "";
    });
  });

  // If the page was re-rendered with previous form values (e.g. after a validation
  // error), restore the markers so the user doesn't have to click the map again.
  (function restoreFromFormValues() {
    const plat = document.getElementById("pickup_lat").value;
    const plon = document.getElementById("pickup_lon").value;
    const dlat = document.getElementById("dropoff_lat").value;
    const dlon = document.getElementById("dropoff_lon").value;
    if (plat && plon) {
      pickupMarker = L.marker([plat, plon], { icon: pickupIcon, draggable: true }).addTo(map);
    }
    if (dlat && dlon) {
      dropoffMarker = L.marker([dlat, dlon], { icon: dropoffIcon, draggable: true }).addTo(map);
      drawRoute();
    }
    if (plat && plon) map.setView([plat, plon], 12);
  })();

  // ---- AJAX submit so the result appears without a full page reload ----
  const form = document.getElementById("fareForm");
  const errorBanner = document.getElementById("errorBanner");
  const errorList = document.getElementById("errorList");
  const successBanner = document.getElementById("successBanner");
  const fareAmount = document.getElementById("fareAmount");
  const fareMeta = document.getElementById("fareMeta");
  const submitBtn = document.getElementById("submitBtn");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = "Predicting…";
    errorBanner.hidden = true;
    successBanner.hidden = true;

    const payload = Object.fromEntries(new FormData(form).entries());

    try {
      const res = await fetch(form.action, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok) {
        errorList.innerHTML = (data.errors || ["Something went wrong."]).map((m) => `<li>${m}</li>`).join("");
        errorBanner.hidden = false;
      } else {
        fareAmount.textContent = `$${data.prediction.toFixed(2)}`;
        fareMeta.textContent = `Estimated trip distance: ${data.computed_distance} km`;
        successBanner.hidden = false;
        successBanner.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    } catch (err) {
      errorList.innerHTML = `<li>Could not reach the server: ${err}</li>`;
      errorBanner.hidden = false;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Predict Fare";
    }
  });
})();
