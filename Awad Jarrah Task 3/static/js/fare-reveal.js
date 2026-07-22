/**
 * fare-reveal.js
 * When the server renders a prediction, count the total up from $0 rather
 * than showing it instantly — a small, purposeful flourish, skipped
 * entirely for anyone with prefers-reduced-motion set.
 */
window.FarePredictor = window.FarePredictor || {};

window.FarePredictor.fareReveal = (function () {
  "use strict";

  function init() {
    var el = document.querySelector("[data-fare]");
    if (!el) return;

    var target = parseFloat(el.getAttribute("data-fare"));
    if (isNaN(target)) return;

    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion) {
      el.textContent = "$" + target.toFixed(2);
      return;
    }

    var start = null;
    var duration = 650;

    function tick(ts) {
      if (start === null) start = ts;
      var progress = Math.min((ts - start) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = "$" + (target * eased).toFixed(2);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);

    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  return { init: init };
})();
