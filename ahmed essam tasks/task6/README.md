# Weather Predictor Pro

This project wraps a pre-trained Simple RNN model for predicting Cairo's daily mean temperature into a modern Django web application. It features a responsive UI with glassmorphism styling and interactive charts.

## Features

- **Single Pipeline Architecture**: Scaler and ML model loaded centrally using a wrapper class for fast inference.
- **Modern UI/UX**: Built with Bootstrap 5, featuring a dynamic gradient background and glass cards.
- **Data Validation**: Ensures exactly 30 numeric temperature values within realistic bounds.
- **Visualization**: Interactive Chart.js sparkline comparing historical input with the prediction.
- **REST API**: Accessible via `/api/predict/` for programmatic JSON integration.

## Installation and Setup

1. **Install Dependencies**:
   Ensure you have Python installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the Model (Optional)**:
   The repository already contains the trained model. If you want to re-train the model:
   ```bash
   python train_and_export.py
   ```

3. **Run the Server**:
   Navigate into the `weather_project` directory and start the Django server:
   ```bash
   cd weather_project
   python manage.py runserver
   ```

4. **View the App**:
   Open a browser and navigate to `http://127.0.0.1:8000/`.


