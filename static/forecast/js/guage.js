(function () {
  function arcPoint(cx, cy, deg, radius) {
    const rad = (deg * Math.PI) / 180;
    return [cx + radius * Math.cos(rad), cy - radius * Math.sin(rad)];
  }

  function buildGauge() {
    const el = document.getElementById('gauge');
    if (!el) return;

    const predicted = parseFloat(el.dataset.predicted);
    const min = parseFloat(el.dataset.min);
    const max = parseFloat(el.dataset.max);

    const cx = 150, cy = 150, r = 120;
    const fraction = Math.max(0, Math.min(1, (predicted - min) / (max - min)));
    const angleDeg = 180 - fraction * 180;

    const [needleX, needleY] = arcPoint(cx, cy, angleDeg, r * 0.82);
    const [startX, startY] = arcPoint(cx, cy, 180, r);
    const [endX, endY] = arcPoint(cx, cy, 0, r);

    const tickLabels = [0, 0.25, 0.5, 0.75, 1]
      .map((f) => {
        const deg = 180 - f * 180;
        const [tx, ty] = arcPoint(cx, cy, deg, r + 16);
        const value = Math.round(min + f * (max - min));
        return `<text x="${tx}" y="${ty}" class="gauge__tick-label" text-anchor="middle" dominant-baseline="middle">${value}</text>`;
      })
      .join('');

    el.innerHTML = `
      <svg viewBox="0 0 300 175" class="gauge__svg">
        <path d="M ${startX} ${startY} A ${r} ${r} 0 0 1 ${endX} ${endY}"
              class="gauge__track" fill="none" />
        <line x1="${cx}" y1="${cy}" x2="${needleX}" y2="${needleY}" class="gauge__needle" />
        <circle cx="${cx}" cy="${cy}" r="6" class="gauge__hub" />
        ${tickLabels}
      </svg>
    `;
  }

  function buildSparkline() {
    const container = document.getElementById('sparkline');
    const dataEl = document.getElementById('sparkline-data');
    if (!container || !dataEl) return;

    const data = JSON.parse(dataEl.textContent);
    if (!data.length) return;

    const values = data.map((d) => d.t2m);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const pad = (max - min) * 0.1 || 1;
    const w = 600, h = 120, m = 8;

    const points = data
      .map((d, i) => {
        const x = m + (i / Math.max(1, data.length - 1)) * (w - m * 2);
        const norm = (d.t2m - (min - pad)) / ((max + pad) - (min - pad));
        const y = h - m - norm * (h - m * 2);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

    container.innerHTML = `
      <svg viewBox="0 0 ${w} ${h}" class="sparkline__svg" preserveAspectRatio="none">
        <polyline points="${points}" class="sparkline__line" fill="none" />
      </svg>
    `;
  }

  document.addEventListener('DOMContentLoaded', function () {
    buildGauge();
    buildSparkline();
  });
})();
