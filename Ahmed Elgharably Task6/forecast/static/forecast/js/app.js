(function () {
  const dataEl = document.getElementById("recent-history-data");
  if (!dataEl || typeof Chart === "undefined") return;

  const rows = JSON.parse(dataEl.textContent);
  const labels = rows.map((r) => r.date.slice(5)); // MM-DD
  const values = rows.map((r) => r.temperature);

  const ctx = document.getElementById("sparkline");
  if (!ctx) return;

  new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          data: values,
          borderColor: "#D98E3F",
          backgroundColor: "rgba(217, 142, 63, 0.15)",
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.35,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: "#9FD8CB", maxTicksLimit: 6, font: { family: "JetBrains Mono", size: 10 } },
          grid: { color: "rgba(159, 216, 203, 0.12)" },
        },
        y: {
          ticks: { color: "#9FD8CB", font: { family: "JetBrains Mono", size: 10 } },
          grid: { color: "rgba(159, 216, 203, 0.12)" },
        },
      },
    },
  });
})();
