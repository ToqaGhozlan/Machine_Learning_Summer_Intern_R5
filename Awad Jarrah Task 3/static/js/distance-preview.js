/**
 * distance-preview.js
 * Computes the same straight-line (haversine) distance the model itself
 * uses, and shows it live in a chip as the user fills in coordinates.
 */
window.FarePredictor = window.FarePredictor || {};

window.FarePredictor.distancePreview = (function () {
  "use strict";

  function toRad(deg) { return (deg * Math.PI) / 180; }

  function haversineKm(lat1, lon1, lat2, lon2) {
    var R = 6371; // matches the exact radius verified against the training data
    var dLat = toRad(lat2 - lat1);
    var dLon = toRad(lon2 - lon1);
    var a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    return R * 2 * Math.asin(Math.sqrt(a));
  }

  function init(form) {
    var pLat = form.querySelector('[name="pickup_latitude"]');
    var pLon = form.querySelector('[name="pickup_longitude"]');
    var dLat = form.querySelector('[name="dropoff_latitude"]');
    var dLon = form.querySelector('[name="dropoff_longitude"]');
    var chip = document.getElementById("distance-chip");
    if (!pLat || !pLon || !dLat || !dLon || !chip) return;

    var valueEl = chip.querySelector(".value");
    var milesEl = chip.querySelector(".miles");

    function refresh() {
      var vals = [pLat, pLon, dLat, dLon].map(function (el) { return parseFloat(el.value); });
      var allPresent = vals.every(function (v) { return !isNaN(v); });
      if (!allPresent) { chip.classList.remove("visible"); return; }

      var km = haversineKm(vals[0], vals[1], vals[2], vals[3]);
      var miles = km * 0.621371;
      valueEl.textContent = km.toFixed(2) + " km";
      milesEl.textContent = "(" + miles.toFixed(2) + " mi)";
      chip.classList.add("visible");
    }

    [pLat, pLon, dLat, dLon].forEach(function (el) {
      el.addEventListener("input", refresh);
    });
    refresh();
  }

  return { init: init };
})();
