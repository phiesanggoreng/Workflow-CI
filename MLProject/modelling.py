"""
Model Training Script for MLflow Project (CI Pipeline)
=======================================================
Author: Nibras Ahmad Badruzzaman
Description: Train RandomForestClassifier with MLflow autolog
             for use in CI/CD pipeline via GitHub Actions.
"""

import os
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "wine_quality_preprocessing")
RANDOM_STATE = 42
CLASS_NAMES = ['class_0', 'class_1', 'class_2']


def load_preprocessed_data():
    """Load preprocessed train and test datasets."""
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).iloc[:, 0].to_numpy(copy=True)
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).iloc[:, 0].to_numpy(copy=True)
    return X_train, X_test, y_train, y_test


def train():
    """Train and log model for CI pipeline using autolog."""
    print("=" * 60)
    print("CI Pipeline - Model Training")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Enable MLflow autologging
    mlflow.autolog()
    
    # Use baseline model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    if mlflow.active_run():
        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")
    print("CI Pipeline training complete!")
    
    return model


if __name__ == "__main__":
    train()
