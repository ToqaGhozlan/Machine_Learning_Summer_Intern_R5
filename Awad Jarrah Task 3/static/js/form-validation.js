/**
 * form-validation.js
 * Soft, non-blocking visual feedback when a coordinate falls outside the
 * plausible NYC range. The server (feature_engineering.py) remains the
 * real source of truth — this is just an early visual hint for the user.
 */
window.FarePredictor = window.FarePredictor || {};

window.FarePredictor.formValidation = (function () {
  "use strict";

  var NYC_BOUNDS = {
    lat: [40.4774, 40.9176],
    lon: [-74.2591, -73.7004],
  };

  function isInRange(value, bounds) {
    return !isNaN(value) && value >= bounds[0] && value <= bounds[1];
  }

  function watch(input, bounds) {
    if (!input) return;
    input.addEventListener("blur", function () {
      if (input.value === "") { input.classList.remove("field-invalid"); return; }
      var ok = isInRange(parseFloat(input.value), bounds);
      input.classList.toggle("field-invalid", !ok);
    });
    input.addEventListener("input", function () {
      input.classList.remove("field-invalid");
    });
  }

  function init(form) {
    watch(form.querySelector('[name="pickup_latitude"]'), NYC_BOUNDS.lat);
    watch(form.querySelector('[name="pickup_longitude"]'), NYC_BOUNDS.lon);
    watch(form.querySelector('[name="dropoff_latitude"]'), NYC_BOUNDS.lat);
    watch(form.querySelector('[name="dropoff_longitude"]'), NYC_BOUNDS.lon);

    // default the datetime field to "now" so the form is never blank
    var dt = form.querySelector('[name="pickup_datetime"]');
    if (dt && !dt.value) {
      var now = new Date(Date.now() - new Date().getTimezoneOffset() * 60000);
      dt.value = now.toISOString().slice(0, 16);
    }
  }

  return { init: init };
})();
