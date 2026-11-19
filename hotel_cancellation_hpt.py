# -*- coding: utf-8 -*-
"""
Hotel Booking Cancellation Prediction
Optuna tuning + MLflow tracking + Streamlit-ready models

Install once (PowerShell):
    pip install optuna category_encoders mlflow

Run the script, then open the MLflow UI from the SAME folder:
    mlflow ui
and go to http://127.0.0.1:5000

IMPORTANT: do not name this file mlflow.py (it would hide the real library).
"""

import os
import json
import time
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cloudpickle
import optuna
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from category_encoders import CountEncoder

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 1. SETTINGS
# =====================================================================
DATA_PATH = r"D:\IRL\06_Machine Learning\Hotel Booking project\Dataset\hotel_bookings.csv"
EXPERIMENT_NAME = "Hotel_Cancellation_Prediction"
RANDOM_STATE = 42
CV_FOLDS = 3                 # folds used inside every Optuna trial
TUNE_SAMPLE_SIZE = 25000     # rows used to tune (keeps tuning fast)
SVM_SAMPLE_SIZE = 15000      # SVM is slow, so it tunes AND trains on this many rows

# n_trials / cv_jobs / dense (Naive Bayes needs dense data) / sample sizes
MODEL_CONFIG = {
    "Logistic_Regression": dict(n_trials=15, cv_jobs=-1, dense=False,
                                tune_rows=TUNE_SAMPLE_SIZE, fit_rows=None),
    "Decision_Tree":       dict(n_trials=20, cv_jobs=-1, dense=False,
                                tune_rows=TUNE_SAMPLE_SIZE, fit_rows=None),
    "Random_Forest":       dict(n_trials=10, cv_jobs=1,  dense=False,
                                tune_rows=TUNE_SAMPLE_SIZE, fit_rows=None),
    "SVM":                 dict(n_trials=8,  cv_jobs=-1, dense=False,
                                tune_rows=SVM_SAMPLE_SIZE, fit_rows=SVM_SAMPLE_SIZE),
    "Naive_Bayes":         dict(n_trials=15, cv_jobs=-1, dense=True,
                                tune_rows=TUNE_SAMPLE_SIZE, fit_rows=None),
}

# =====================================================================
# 2. DATA CLEANING (row filtering only; column work lives in the pipeline)
# =====================================================================
df = pd.read_csv(DATA_PATH)

# Leakage / unusable columns:
#  - reservation_status(_date): reveal the outcome
#  - assigned_room_type: only known at check-in (leaks the target)
#  - company: ~94% missing
df = df.drop(columns=["company", "assigned_room_type",
                      "reservation_status", "reservation_status_date"])
df = df.drop_duplicates()

# Remove invalid / extreme adr values (negative and the ~5400 outlier)
df = df[(df["adr"] >= 0) & (df["adr"] <= 1000)]

print("Cleaned data shape:", df.shape)


# =====================================================================
# 3. FEATURE ENGINEERING (inside the model, so Streamlit can send RAW input)
# =====================================================================
def _agent_to_str(a):
    """agent is an ID, so treat it as a category (string), not a number."""
    if pd.isna(a):
        return "Unknown"
    try:
        return str(int(float(a)))
    except (TypeError, ValueError):
        return str(a)


def add_features(X):
    """Takes the raw booking columns and returns the model-ready columns.
    Every feature is row-wise, so there is no data leakage."""
    X = X.copy()

    # ---- basic cleaning ----
    X["children"] = X["children"].fillna(0)
    X["country"] = X["country"].fillna("Unknown")
    X["agent"] = X["agent"].apply(_agent_to_str)

    # ---- requested features ----
    X["total_guests"] = X["adults"] + X["children"] + X["babies"]
    X["total_nights"] = X["stays_in_weekend_nights"] + X["stays_in_week_nights"]
    X["total_previous_bookings"] = (
        X["previous_cancellations"] + X["previous_bookings_not_canceled"]
    )
    X["previous_cancel_rate"] = (
        X["previous_cancellations"] /
        X["total_previous_bookings"].replace(0, np.nan)
    ).fillna(0)
    X["adr_per_guest"] = (
        X["adr"] / X["total_guests"].replace(0, np.nan)
    ).fillna(0)
    X["is_family"] = ((X["children"] + X["babies"]) > 0).astype(int)
    X["is_weekend_stay"] = (X["stays_in_weekend_nights"] > 0).astype(int)
    X["has_previous_booking"] = (X["total_previous_bookings"] > 0).astype(int)

    # ---- extra features ----
    X["total_stay_cost"] = X["adr"] * X["total_nights"]
    X["has_special_requests"] = (X["total_of_special_requests"] > 0).astype(int)
    X["has_parking"] = (X["required_car_parking_spaces"] > 0).astype(int)

    month_map = {m: i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"], start=1)}
    arrival_date = pd.to_datetime(
        dict(year=X["arrival_date_year"],
             month=X["arrival_date_month"].map(month_map),
             day=X["arrival_date_day_of_month"]),
        errors="coerce",
    )
    X["is_weekend_arrival"] = arrival_date.dt.day_of_week.isin([5, 6]).astype(int)
    X["is_peak_season"] = X["arrival_date_month"].isin(
        ["June", "July", "August"]).astype(int)

    # The data only covers 2015-2017, so the year would not generalise
    return X.drop(columns=["arrival_date_year"])


# =====================================================================
# 4. TRAIN / TEST SPLIT (on RAW columns)
# =====================================================================
X = df.drop(columns="is_canceled")
y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

ONEHOT_COLS = [
    "hotel", "meal", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type", "arrival_date_month",
]
FREQ_COLS = ["country", "agent"]

_featured = add_features(X.head(2000))
NUM_COLS = _featured.select_dtypes(include=np.number).columns.to_list()

# Safety check: every engineered column must be handled by the encoders
assert set(NUM_COLS) | set(ONEHOT_COLS) | set(FREQ_COLS) == set(_featured.columns), \
    "Column lists are out of sync with add_features()"


def make_preprocessor(dense=False):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ])
    freq_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", CountEncoder(normalize=True, handle_unknown="value",
                                 handle_missing="value")),
    ])
    return ColumnTransformer(
        [("num", num_pipeline, NUM_COLS),
         ("ohe", onehot_pipeline, ONEHOT_COLS),
         ("freq", freq_pipeline, FREQ_COLS)],
        remainder="drop",
    )


# =====================================================================
# 5. MODELS + OPTUNA SEARCH SPACES
# =====================================================================
def suggest_params(trial, name):
    if name == "Logistic_Regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if name == "Decision_Tree":
        return {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if name == "Random_Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced_subsample"]),
        }
    if name == "SVM":
        return {
            "C": trial.suggest_float("C", 0.1, 20, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 1e-1, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if name == "Naive_Bayes":
        return {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-11, 1.0, log=True),
        }
    raise ValueError(f"Unknown model: {name}")


def make_estimator(name, p):
    if name == "Logistic_Regression":
        return LogisticRegression(C=p["C"], class_weight=p["class_weight"],
                                  max_iter=1000, random_state=RANDOM_STATE)
    if name == "Decision_Tree":
        return DecisionTreeClassifier(
            criterion=p["criterion"], max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            class_weight=p["class_weight"], random_state=RANDOM_STATE)
    if name == "Random_Forest":
        return RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"], class_weight=p["class_weight"],
            n_jobs=-1, random_state=RANDOM_STATE)
    if name == "SVM":
        return SVC(kernel="rbf", C=p["C"], gamma=p["gamma"],
                   class_weight=p["class_weight"], random_state=RANDOM_STATE)
    if name == "Naive_Bayes":
        return GaussianNB(var_smoothing=p["var_smoothing"])
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(name, params):
    """raw booking columns -> features -> encoding/scaling -> classifier"""
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preprocessor", make_preprocessor(dense=MODEL_CONFIG[name]["dense"])),
        ("classifier", make_estimator(name, params)),
    ])


def make_objective(name, X_tune, y_tune):
    cfg = MODEL_CONFIG[name]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        pipe = build_pipeline(name, suggest_params(trial, name))
        return cross_val_score(pipe, X_tune, y_tune, scoring="f1",
                               cv=skf, n_jobs=cfg["cv_jobs"]).mean()

    return objective


def trial_logger(name):
    """Logs every Optuna trial as a child run under the model's parent run."""
    def _callback(study, trial):
        if trial.value is None:
            return
        with mlflow.start_run(run_name=f"{name}_trial_{trial.number}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric("cv_f1", trial.value)
            mlflow.set_tag("trial_of", name)
    return _callback


def sample_rows(X_, y_, n):
    if n is None or n >= len(X_):
        return X_, y_
    X_s, _, y_s, _ = train_test_split(X_, y_, train_size=n, stratify=y_,
                                      random_state=RANDOM_STATE)
    return X_s, y_s


def evaluate(pipe, X_te, y_te):
    y_pred = pipe.predict(X_te)
    try:
        score = pipe.predict_proba(X_te)[:, 1]
    except (AttributeError, NotImplementedError):
        score = pipe.decision_function(X_te)
    return {
        "test_accuracy": accuracy_score(y_te, y_pred),
        "test_precision": precision_score(y_te, y_pred),
        "test_recall": recall_score(y_te, y_pred),
        "test_f1": f1_score(y_te, y_pred),
        "test_roc_auc": roc_auc_score(y_te, score),
    }


# =====================================================================
# 6. INPUT SCHEMA FOR STREAMLIT (dropdown options + number ranges)
# =====================================================================
numeric_raw_cols = [c for c in X.select_dtypes(include=np.number).columns
                    if c not in ("agent", "arrival_date_year")]
input_schema = {
    "input_columns": list(X.columns),
    "categorical_options": {c: sorted(X[c].dropna().unique().tolist())
                            for c in ONEHOT_COLS + ["country"]},
    "agent_options": sorted(X["agent"].dropna().astype(int).unique().tolist()),
    "arrival_year_options": sorted(X["arrival_date_year"].unique().tolist()),
    "numeric_ranges": {c: [float(X[c].min()), float(X[c].max())]
                       for c in numeric_raw_cols},
}
with open("input_schema.json", "w") as f:
    json.dump(input_schema, f, indent=2)


# =====================================================================
# 7. TUNE + TRAIN + LOG EVERY MODEL TO MLFLOW
# =====================================================================
from pathlib import Path
mlflow.set_tracking_uri(f"sqlite:///{Path(__file__).resolve().parent.as_posix()}/mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

# A few raw rows (no missing values) so MLflow can store the input signature
input_example = X_test.head(5).copy()
input_example["children"] = input_example["children"].fillna(0)
input_example["country"] = input_example["country"].fillna("Unknown")
input_example["agent"] = input_example["agent"].fillna(0)

results = []
fitted_models = {}

for model_id, (name, cfg) in enumerate(MODEL_CONFIG.items()):
    print(f"\n=== {name}: tuning ({cfg['n_trials']} trials) ===")

    with mlflow.start_run(run_name=name) as run:
        # ---- Optuna search ----
        X_tune, y_tune = sample_rows(X_train, y_train, cfg["tune_rows"])
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        t0 = time.time()
        study.optimize(make_objective(name, X_tune, y_tune),
                       n_trials=cfg["n_trials"], callbacks=[trial_logger(name)])
        tuning_time = time.time() - t0

        best_params = study.best_params
        print(f"Best CV F1: {study.best_value:.4f} | params: {best_params}")

        # ---- final fit with the best parameters ----
        X_fit, y_fit = sample_rows(X_train, y_train, cfg["fit_rows"])
        pipe = build_pipeline(name, best_params)

        t0 = time.time()
        pipe.fit(X_fit, y_fit)
        train_time = time.time() - t0

        t0 = time.time()
        metrics = evaluate(pipe, X_test, y_test)
        test_time = time.time() - t0

        train_pred = pipe.predict(X_fit)
        metrics["train_accuracy"] = accuracy_score(y_fit, train_pred)
        metrics["train_f1"] = f1_score(y_fit, train_pred)
        metrics["cv_f1"] = study.best_value
        metrics["tuning_time"] = tuning_time
        metrics["train_time"] = train_time
        metrics["test_time"] = test_time
        metrics["model_size_bytes"] = len(cloudpickle.dumps(pipe))
        metrics["model_id"] = model_id

        # ---- log to MLflow ----
        mlflow.log_params(best_params)
        mlflow.log_params({
            "n_trials": cfg["n_trials"],
            "cv_folds": CV_FOLDS,
            "tuning_rows": len(X_tune),
            "fit_rows": len(X_fit),
            "optimized_metric": "f1",
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_name", name)
        mlflow.log_dict(input_schema, "input_schema.json")

        model_artifact = f"{name}_model"
        try:
            mlflow.sklearn.log_model(
                sk_model=pipe, name=model_artifact,
                serialization_format="cloudpickle",
                input_example=input_example)
        except Exception as e:  # never lose a finished model over the signature
            print(f"  (signature skipped for {name}: {type(e).__name__})")
            mlflow.sklearn.log_model(
                sk_model=pipe, name=model_artifact,
                serialization_format="cloudpickle")

        print(f"{name}: test F1={metrics['test_f1']:.4f} | "
              f"test acc={metrics['test_accuracy']:.4f} | "
              f"train acc={metrics['train_accuracy']:.4f} | "
              f"tuning {tuning_time:.0f}s")

        results.append({"model": name, "run_id": run.info.run_id,
                        "model_uri": f"runs:/{run.info.run_id}/{model_artifact}",
                        "best_params": best_params, **metrics})
        fitted_models[name] = pipe


# =====================================================================
# 8. SUMMARY, BEST MODEL TAG, FILES FOR STREAMLIT
# =====================================================================
summary = pd.DataFrame(results).sort_values("test_f1", ascending=False)
cols = ["model", "cv_f1", "test_f1", "test_accuracy", "test_precision",
        "test_recall", "test_roc_auc", "train_accuracy", "train_time", "model_size_bytes"]
print("\n--- Model comparison (best first) ---")
print(summary[cols].round(4).to_string(index=False))
summary.drop(columns=["best_params"]).to_csv("model_comparison.csv", index=False)

best_name = summary.iloc[0]["model"]
best_run_id = summary.iloc[0]["run_id"]
MlflowClient().set_tag(best_run_id, "best_model", "true")
print(f"\nBest model by test F1: {best_name}  (run_id: {best_run_id})")

# Self-contained copy of the best model (cloudpickle stores add_features inside)
with open("best_model.pkl", "wb") as f:
    cloudpickle.dump(fitted_models[best_name], f)

print("Saved: best_model.pkl, input_schema.json, model_comparison.csv")
print("Next: run  mlflow ui  in this folder and compare the runs.")
