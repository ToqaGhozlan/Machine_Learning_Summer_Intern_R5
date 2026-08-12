# Task 5 forecasting summary

- Dataset: Cairo
- Frequency: D
- Target variable: temperature
- Seasonal period used: 7
- Missing timestamps detected after resampling: 0
- Missing values before preprocessing: 0
- Missing values after preprocessing: 0
- Outlier report: flagged=3, modified=3
- Outlier stats: before min=-2.250, before max=25.930; after min=-2.250, after max=25.930
- Near-zero test values for MAPE stability: 0; minimum absolute actual test value: 0.200

## Stationarity
- ADF on training target: statistic=-1.277, p-value=0.640
- KPSS on training target: statistic=1.347, p-value=0.010
- Differencing decision: first differencing was used because the training series was non-stationary; seasonal differencing was not used because the daily series did not show a strong seasonal unit root signal.

## Model selection
- Manual ACF/PACF candidate: (2, 1, 1) with seasonal order (0, 0, 0, 7)
- Auto_arima selected: order=(2, 1, 1), seasonal_order=(0, 0, 0, 7)
- Validation-selected SARIMA order: order=(2, 1, 2), seasonal_order=(0, 0, 0, 7), AIC=1067.76
- Validation metrics used for selection: MAE=3.385, RMSE=4.181

### SARIMA candidate comparison
    order seasonal_order         aic      mae     rmse      mape     smape
(1, 1, 1)   (0, 0, 0, 7) 1084.910448 3.656242 4.589576 50.328852 33.886693
(2, 1, 1)   (0, 0, 0, 7) 1071.315980 3.442111 4.277072 46.769838 32.540294
(1, 1, 2)   (0, 0, 0, 7) 1066.989992 3.513636 4.392085 48.060892 32.982227
(2, 1, 2)   (0, 0, 0, 7) 1067.764796 3.385071 4.181388 45.663785 32.187762
(1, 1, 1)   (1, 0, 0, 7) 1059.859248 3.584284 4.495822 49.257297 33.420316
(1, 1, 1)   (0, 0, 1, 7) 1056.025567 3.628383 4.550558 49.881087 33.713138
(2, 1, 1)   (1, 0, 1, 7) 1047.042393 3.456484 4.302531 47.062468 32.623783
(2, 1, 2)   (1, 0, 1, 7) 1046.557970 3.448032 4.291480 46.932865 32.567127

## Index alignment checks
- ARIMA forecast aligned to test index: True
- SARIMA forecast aligned to test index: True
- Persistence baseline aligned to test index: True
- Seasonal naive baseline aligned to test index: True

## Diagnostics
- Ljung-Box p-value for residual autocorrelation: 0.4970
- A high p-value supports the absence of significant residual autocorrelation.

## Test-set metrics
               Model      mae     rmse       mape      smape
Persistence baseline 4.070714 4.483082 276.742532 107.178497
      Seasonal naive 4.637857 5.592218 208.040994 129.055733
               ARIMA 4.093025 4.768215 342.563211  98.408075
              SARIMA 4.065185 4.722809 338.677585  98.468434

## Walk-forward validation
 fold  train_end validation_start validation_end      mae     rmse       mape      smape
    1 2022-12-10       2022-12-11     2022-12-17 1.635283 1.803263 163.341383 182.246723
    2 2022-12-17       2022-12-18     2022-12-24 7.652076 8.115892 103.670703 192.955129
    3 2022-12-24       2022-12-25     2022-12-31 1.879720 2.343676  20.518702  22.637696

## Conclusion
The Task 5 workflow was implemented on the existing Task 4 weather dataset using a verified daily frequency and a seasonal period of 7. Missing timestamps were detected after reindexing, short gaps were interpolated, and long gaps were filled with a seasonal-naive approach. The pipeline uses a chronological split, a conservative outlier policy, and a validation-based SARIMA selection strategy. MAPE is reported for completeness but is less stable when temperature values are near zero; MAE and RMSE are treated as the primary metrics. SARIMAX was not used because the available exogenous variables were not available at forecast time without introducing leakage.