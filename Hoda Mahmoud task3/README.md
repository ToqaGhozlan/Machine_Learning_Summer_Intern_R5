# Uber Fare Prediction Deployment

This repository contains the notebook work for Task 1 and Task 2 plus a deployable Task 3 Flask app for Uber fare prediction.

## Project Structure

- `Task1.ipynb` - exploratory data analysis notebook.
- `Task2.ipynb` - preprocessing and model experimentation notebook.
- `uber_fare_pipeline.py` - shared feature engineering and input validation.
- `train_and_export.py` - training, model comparison, tuning, and export script.
- `app.py` - Flask application.
- `templates/index.html` - Flask UI template.
- `static/style.css` - Flask styling.
- `requirements.txt` - Python dependencies.

## Dataset

The project is built for the NYC Uber fare regression dataset. The target column is `Fare_Amount`. The deployment pipeline expects raw trip details such as pickup/dropoff coordinates, passenger count, pickup time, and categorical trip context.

## Task 1 and Task 2 Notes

The existing notebooks already cover the main analysis and preprocessing ideas, but they mix several column-name conventions. For deployment, the shared pipeline in `uber_fare_pipeline.py` normalizes the raw input schema so training and Flask inference use the same feature definitions.

## Model Comparison

`train_and_export.py` compares at least three regression models:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

It reports `MAE`, `RMSE`, and `R2` on a held-out test split, then tunes XGBoost with `RandomizedSearchCV`.

## Final Model

The script selects the best model by test-set `RMSE` and saves the full fitted pipeline with `joblib.dump(...)` to `uber_fare_model.pkl`.

## Preprocessing Pipeline

The saved pipeline performs:

1. Raw trip validation in the Flask app.
2. Feature engineering from `Pickup_Datetime` and coordinates.
3. Numeric imputation and scaling.
4. Categorical imputation and one-hot encoding.
5. Final regression model prediction.

The same pipeline is used during training and inference.

## Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Train and Save the Model

Place `final_internship_data.csv` in the project root, then run:

```bash
python train_and_export.py --data final_internship_data.csv
```

This creates:

- `uber_fare_model.pkl`
- `model_comparison.csv`

## Run Flask

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## How to Test

Use the form to submit at least three valid trips:

1. Short trip.
2. Medium-distance trip.
3. Longer trip.

Also test one invalid submission, such as a missing field or an invalid coordinate, and confirm the app shows a clear validation message.

## Known Limitations

- The repository currently does not include the dataset file itself, so the model artifact must be generated locally before the app can predict.
- Task 1 and Task 2 notebooks contain legacy column-name inconsistencies; the shared training pipeline resolves the deployment-side schema.

## What Could Be Improved Next

- Add automated Flask route tests.
- Store the model comparison table as a notebook output as well as CSV.
- Add stronger outlier handling or target transformation if needed after training on the full dataset.