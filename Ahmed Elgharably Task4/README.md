# Task 4 — Weather Forecast API & Time-Series EDA

## Important — this one needs to run on your machine

My sandbox can't reach `power.larc.nasa.gov` (it's outside my allowed network egress,
and the page also blocks the kind of automated fetch my browsing tool uses), so I
could not execute this notebook myself and pull real numbers. Everything below is
real, working code — I syntax-checked every cell and ran the full pipeline end-to-end
against synthetic weather-shaped data to catch bugs (see "What I verified" below) —
but you need to run it yourself against the live API to get your actual results and
plots.

## What's in this folder

| File | Purpose |
|---|---|
| `Task4_Weather_TimeSeries_EDA.ipynb` | Full notebook: Part 1 (API pull), Part 2 (cleaning), Part 3.1–3.4 (EDA), written summary template. |
| `requirements.txt` | Dependencies. |
| `data/` | Where the notebook saves the raw API response and cleaned CSV when you run it. |

## Region & API choice (already decided in the notebook, can be changed)

- **Region:** Damietta, Egypt (31.4165° N, 31.8133° E) — change `LAT`/`LON` near the
  top of Part 1 if you'd rather use a different city/governorate.
- **API:** [NASA POWER](https://power.larc.nasa.gov/), `temporal/daily/point`
  endpoint. **No API key or signup needed** — this was the deciding factor over
  OpenWeatherMap/Visual Crossing, which both require registering for a key.
- **Time range:** two full calendar years, 2024–2025 (731 days), not just one. This
  isn't arbitrary — `statsmodels.seasonal_decompose` with an annual period needs
  **two complete cycles (≥730 daily observations)** to run at all; requesting only
  one year (365 days) makes that cell crash with a `ValueError`. Pulling two years
  fixes that and gives the trend/rolling-mean plots something real to compare across.

## How to run it

```bash
pip install -r requirements.txt
jupyter notebook Task4_Weather_TimeSeries_EDA.ipynb
```

Run all cells top to bottom. No API key, `.env` file, or config needed — the first
code cell in Part 1 calls the API directly. It should complete in well under a
minute (NASA POWER's own docs note point/single-location requests are their fastest
tier).

## What I verified without real data

Since I couldn't hit the live endpoint, I replaced the API-call cell with synthetic
two-year daily data shaped like a real Mediterranean coastal climate (seasonal
temperature swing, winter-concentrated rainfall, a touch of noise and a slight
warming drift) and ran every remaining cell against it end-to-end. That caught and
fixed one real bug before it reached you: `seasonal_decompose(period=365)` needs
≥730 observations, which a literal "one year" reading of the task would have violated
— this is why the notebook pulls 2024–2025 instead of just 2025.

What I could **not** verify, because it depends on the real API response shape and
your actual region's numbers:
- Whether NASA POWER returns any `-999` sentinel gaps for this specific point/date
  range (the cleaning cell handles it either way, but I haven't seen it happen live).
- The actual ADF p-value / stationarity conclusion, actual ACF/PACF shape, and actual
  correlation numbers — the markdown cells describe the *expected* direction based on
  Damietta's known climate, but you should read the real output and adjust the
  written-summary section if reality differs from the prediction.

## Filling in the written summary

The last cell in the notebook is a half-page summary with the structure already
written and bracketed placeholders — `[Fill in from 3.2/3.3: ...]` — for the parts
that depend on your actual run's numbers/plots. Replace those after running.
