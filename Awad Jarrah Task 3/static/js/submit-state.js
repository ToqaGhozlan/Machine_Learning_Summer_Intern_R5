/**
 * submit-state.js
 * Puts the submit button into a visible "working" state the instant the
 * form is submitted, so there's never a dead pause after clicking.
 */
window.FarePredictor = window.FarePredictor || {};

window.FarePredictor.submitState = (function () {
  "use strict";

  function init(form) {
    var btn = form.querySelector(".submit-btn");
    if (!btn) return;

    form.addEventListener("submit", function () {
      btn.disabled = true;
      var spinner = btn.querySelector(".spinner");
      var label = btn.querySelector(".btn-label");
      if (spinner) spinner.classList.add("visible");
      if (label) label.textContent = "Calculating fare…";
    });
  }

  return { init: init };
})();
