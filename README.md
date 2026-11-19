# Hotel Booking Cancellation Prediction

A machine learning web application that predicts whether a hotel booking is likely to be cancelled. The application uses a trained scikit-learn pipeline and Streamlit to collect booking details and display a prediction.

## Project Overview

This project uses the Hotel Booking Demand dataset and includes:

- Data cleaning and feature engineering
- Categorical encoding and numerical feature scaling
- Model training and hyperparameter tuning with Optuna
- Experiment tracking with MLflow
- A Streamlit interface for making predictions with the saved model

The training workflow evaluates five classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. Gaussian Naive Bayes

The training script selects the best model based on test F1 score and saves the fitted preprocessing-and-model pipeline as `best_model.pkl`.

## Repository Structure

```text
Hotel-Cancellation-Prediction/
├── app.py
├── final_model.pkl
├── input_schema.json
├── requirements.txt
├── hotel_cancellation_hpt.py
└── README.md
```

The Streamlit application requires `app.py`, a compatible trained model file (`final_model.pkl` or `best_model.pkl`), and `input_schema.json`.

> The MLflow database and `mlruns/` tracking directory are not required to run the Streamlit prediction application. They may be retained separately for experiment tracking.

## Tech Stack

- Python 3.11
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Category Encoders
- Optuna
- MLflow
- Cloudpickle

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Replace the URL with your GitHub repository URL.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For deployment, `requirements.txt` should contain the runtime dependencies needed by `app.py`. The training script additionally uses Optuna and MLflow.

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How to Use the Application

1. Open the Streamlit application.
2. Enter the hotel, arrival, guest, stay, and booking-history details.
3. Select **Predict Cancellation**.
4. Review the predicted booking outcome and cancellation probability when the loaded model supports probability predictions.

The prediction is an estimate from the trained model and is not a guarantee of the actual booking outcome.

## Model Training and Experiment Tracking

The training script `hotel_cancellation_hpt.py` performs the following workflow:

1. Loads and cleans the hotel booking dataset.
2. Engineers booking-related features inside the model pipeline.
3. Splits the data into training and test sets.
4. Tunes model hyperparameters using Optuna with stratified cross-validation and F1 scoring.
5. Trains each model using its selected hyperparameters.
6. Evaluates model performance using accuracy, precision, recall, F1 score, and ROC-AUC.
7. Logs parameters, metrics, and model artifacts to MLflow.
8. Saves the model comparison results and the best fitted pipeline.

### Training dependencies

The training script uses additional packages, including:

```bash
pip install optuna mlflow category-encoders
```

Before running the training script, update `DATA_PATH` in `hotel_cancellation_hpt.py` to the location of your local `hotel_bookings.csv` dataset.

Then run:

```bash
python hotel_cancellation_hpt.py
```

The script generates files such as:

- `best_model.pkl`
- `input_schema.json`
- `model_comparison.csv`
- MLflow tracking data

## Deploy on Streamlit Community Cloud

1. Push the application files, model file, schema, and `requirements.txt` to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select the GitHub repository and branch.
4. Set the main file path to `app.py`.
5. Select a Python version compatible with the environment used to train and serialize the model.
6. Deploy the application.

Keep the model file and `input_schema.json` in the same repository directory as `app.py`, unless the paths in the application are changed.

## Important Notes

- The saved model must be the complete fitted pipeline, including feature engineering, preprocessing, and classifier. A classifier-only file will not accept the raw booking fields expected by the app.
- Python and package versions should be compatible with those used when the model was trained and serialized.
- Do not commit virtual environments, dataset files, secrets, or unnecessary MLflow tracking directories to the deployment repository.
- Only upload datasets when you have permission to share them.

## Future Improvements

- Add batch prediction using CSV uploads.
- Add model-performance visualizations.
- Improve input validation and user guidance.
- Add automated tests for the prediction pipeline.
- Monitor model performance as new booking data becomes available.

## Author

SAIRAM PAMULA
