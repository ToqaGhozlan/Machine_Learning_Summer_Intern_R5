# Put your Task 5 model files here

Copy these three files (produced by your Task 5 notebook's pickle-saving cell)
into this folder:

- `arma_manual.pkl`
- `armax_auto.pkl`
- `fourier_auto.pkl`

The Django app loads them once, at server startup, from this exact folder
(`predictor/apps.py` -> `ml_models.load_all_models()`), so the app must be
restarted after you add or replace a file here.
