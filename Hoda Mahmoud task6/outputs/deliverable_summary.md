Region and time range
---------------------
The deployed Task 5 model uses London, United Kingdom weather data (requested point: latitude 51.5074, longitude -0.1278; NASA POWER resolved point: latitude 51.507, longitude -0.128). The dataset covers 2022-01-01 through 2022-12-31 (daily observations).

API and justification
---------------------
Data were retrieved from the NASA POWER daily point API (https://power.larc.nasa.gov/api/temporal/daily/point). I chose NASA POWER because it provides free, stationless daily historical meteorological variables by latitude/longitude for long ranges without requiring an API key, making the workflow reproducible.

Main temporal patterns
---------------------
- Temperature: daily mean temperature `T2M` shows clear seasonal cycles (higher values in mid-year, lower in winter). A 30-day rolling mean highlights a smooth seasonal trend; STL decomposition (period 365) separates a dominant seasonal component and a slowly-varying trend. Local volatility (30-day rolling std) is modest compared with the seasonal amplitude.
- Precipitation: `PRECTOT` is heavily zero-inflated (many dry days) with sporadic rainfall events. Monthly boxplots show heavier tails during typical wet months. Correlation between daily temperature and precipitation is low (see correlation matrix).

Stationarity
------------
An Augmented Dickey–Fuller (ADF) test was applied to the temperature series: ADF statistic ≈ -1.718, p-value ≈ 0.422 (fail to reject non-stationarity at 5% level). This matches expectations because the temperature series contains a strong seasonal component. For modeling, detrending and/or seasonal differencing (or seasonal decomposition) will be required to achieve stationarity.

Data quality issues and handling
-----------------------------
- Missing values: counts were computed per column. Temperature fields had no missing values after the reindex; precipitation originally had gaps which were filled with zeros by policy in this script (reasonable for short gaps in dry climates but should be reviewed for long gaps).
- Units: NASA POWER supplies temperatures in °C and precipitation in mm/day — units are consistent in the cleaned CSV.
- Suspicious values: no extreme impossible temperature spikes were present in the retrieved period; precipitation contains many zeros by design. Any flagged values or long gaps would be logged and require manual inspection.

Files produced
--------------
- Raw API response: outputs/raw/power_response.json
- Cleaned CSV: outputs/cleaned/cleaned_weather.csv
- EDA artifacts (plots, decomposition, ACF/PACF, summary stats): outputs/eda/
- Short summary: outputs/summary.md and this deliverable (outputs/deliverable_summary.md)

If you want, I can convert the script into a Jupyter notebook with inline visual narrative and expanded commentary for submission.
