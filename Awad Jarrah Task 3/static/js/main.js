/**
 * main.js
 * Single entry point — wires up each independent module in
 * static/js/*.js once the DOM is ready. Each module only touches its
 * own piece of the page, so they can be edited or removed independently.
 */
document.addEventListener("DOMContentLoaded", function () {
  var form = document.querySelector("form");
  var FP = window.FarePredictor;

  if (form && FP) {
    FP.distancePreview && FP.distancePreview.init(form);
    FP.formValidation && FP.formValidation.init(form);
    FP.submitState && FP.submitState.init(form);
  }
  FP.fareReveal && FP.fareReveal.init();
});
