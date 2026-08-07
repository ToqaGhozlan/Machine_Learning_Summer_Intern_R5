## Final Summary

**Region:** Algiers, Algeria (36.75°N, 3.06°E), a coastal Mediterranean-climate city.

**API and rationale:** NASA POWER's `temporal/daily/point` endpoint — chosen because it
requires no registration or API key, has no meaningful rate-limit friction for a single-point
multi-year pull, and returns temperature, precipitation, humidity, wind, and cloud cover in
one call, which covers every variable the assignment asks for without needing a second source.

**Time range analyzed:** 2024-01-01 through 2025-12-31 — two full calendar years of daily
data (730 expected days), long enough to observe two complete seasonal cycles and give
`seasonal_decompose`/STL a reliable period-365 estimate.

**Main temporal patterns found:** a strong, regular annual temperature cycle (trough in
Jan/Feb, peak in Jul/Aug) with a mild multi-month warming trend (~+0.3-0.4°C/year on the
30-day rolling mean) superimposed on it; local volatility (30-day rolling std) that rises in
the spring/autumn shoulder seasons and settles down at the height of summer; and a classic
Mediterranean wet-winter (Nov-Feb)/dry-summer (Jun-Aug) precipitation regime confirmed by
both the raw time series and the monthly boxplots. ACF/PACF show strong short-lag
persistence (PACF cuts off after lag ~1-2) riding on top of the annual periodicity visible
in the ACF's slow, wave-like decay.

**Is the series stationary?** No — the ADF test on the raw daily temperature series fails to
reject the unit-root null (p ≈ 0.67), so it is classified as non-stationary. This is expected
given how large and regular the annual seasonal swing is relative to short-run noise, since
ADF's test model doesn't separately account for deterministic seasonality. Running ADF again
on the STL residual (seasonality and trend removed) confirms the diagnosis by returning a
much lower p-value, i.e. the series becomes stationary once the seasonal component is taken
out. Practical implication: a forecasting model built on this data should use seasonal
differencing or an explicitly seasonal model (SARIMA, or modeling the STL residual directly)
rather than plain first-order differencing.

**Data quality issues found and how they were handled:** (1) NASA POWER's `-999` missing-value
sentinel appeared in temperature (14 days) and precipitation (18 days) — converted to `NaN`,
then temperature/humidity/wind/cloud were time-interpolated (physically smooth variables)
while precipitation gaps were filled with 0 (conservative default for an erratic variable, and
this dataset's specific gaps fell in low-rain-probability periods). (2) Three duplicated
timestamps — resolved by keeping the first occurrence. (3) Five missing calendar days
(gaps in the sampling frequency) — resolved by reindexing onto the complete daily calendar so
gaps are explicit `NaN` rather than silently absent, then imputed with the same strategy.
(4) Two impossible sensor/API-error outliers — a 55.4°C temperature reading and a -12 mm
precipitation reading — identified via 30-day rolling z-score (local, season-aware) and a
physical-impossibility check (negative rainfall) respectively, distinguished from real
weather extremes by cross-checking against Algiers' known historical temperature range and
rainfall's hard physical lower bound, then corrected via the same interpolation/zero-fill
approach used for the missing-value pass.
