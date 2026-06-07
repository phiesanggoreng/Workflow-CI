"""
Model Training Script for MLflow Project (CI Pipeline)
=======================================================
Author: Nibras Ahmad Badruzzaman
Description: Train RandomForestClassifier with MLflow logging
             for use in CI/CD pipeline via GitHub Actions.
"""

import os
import json
import subprocess
import urllib.request
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "wine_quality_preprocessing")
EXPERIMENT_NAME = "Wine_Quality_CI"
RANDOM_STATE = 42
CLASS_NAMES = ['class_0', 'class_1', 'class_2']


def ensure_local_mlflow_server():
    """Start a local MLflow server if one is not already running."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri and tracking_uri.startswith(("http://", "https://")):
        return tracking_uri

    server_url = "http://127.0.0.1:5000"
    try:
        with urllib.request.urlopen(f"{server_url}/health", timeout=2):
            return server_url
    except Exception:
        pass

    db_path = os.path.join(os.path.dirname(__file__), "mlflow.db")
    artifact_root = os.path.join(os.path.dirname(__file__), "mlartifacts")
    os.makedirs(artifact_root, exist_ok=True)

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            "5000",
            "--backend-store-uri",
            f"sqlite:///{db_path}",
            "--default-artifact-root",
            artifact_root,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{server_url}/health", timeout=2):
                return server_url
        except Exception:
            pass
        import time

        time.sleep(1)

    raise RuntimeError("Local MLflow server did not start on http://127.0.0.1:5000")


def load_preprocessed_data():
    """Load preprocessed train and test datasets."""
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).iloc[:, 0].to_numpy(copy=True)
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).iloc[:, 0].to_numpy(copy=True)
    return X_train, X_test, y_train, y_test


def train():
    """Train and log model for CI pipeline."""
    print("=" * 60)
    print("CI Pipeline - Model Training")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Use DagsHub when configured, otherwise bootstrap a local MLflow server.
    tracking_uri = ensure_local_mlflow_server()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {tracking_uri}")
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nBest Params: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    
    with mlflow.start_run(run_name="CI_RandomForest"):
        # Log params
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Log model (save locally then log as run artifact directory)
        model_dir = os.path.join("ci_artifacts", "model")
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
        mlflow.sklearn.save_model(best_model, model_dir)
        mlflow.log_artifacts(model_dir, artifact_path="model")
        
        # Log confusion matrix
        artifacts_dir = "ci_artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        mlflow.log_artifact(cm_path)
        
        # Log classification report
        report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
        report_path = os.path.join(artifacts_dir, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        
        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print("CI Pipeline training complete!")
    
    return best_model


if __name__ == "__main__":
    train()
