"""
model_utils.py — Machine Learning Utilities for MaterialMind AI

This module provides common utility functions for:
1. Loading processed feature datasets.
2. Group-based train/test splitting (GroupShuffleSplit) to strictly prevent data leakage
   across repeated tests of the same alloy grade.
3. Defining strength classification categories based on physical thresholds.
4. Calculating standardized regression and classification evaluation metrics.
5. Saving and loading models with metadata dictionaries using Joblib.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")


def load_dataset(dataset_name="steel_db"):
    """
    Load the feature-engineered dataset by name.

    Parameters
    ----------
    dataset_name : str
        'steel_db' or 'matnavi'

    Returns
    -------
    pd.DataFrame
    """
    if dataset_name == "steel_db":
        path = os.path.join(PROCESSED_DIR, "steel_db_features.csv")
    elif dataset_name == "matnavi":
        path = os.path.join(PROCESSED_DIR, "matnavi_features.csv")
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Expected 'steel_db' or 'matnavi'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at: {path}")

    df = pd.read_csv(path)
    return df


def split_data_grouped(df, feature_cols, target_col, group_col, test_size=0.2, random_state=42):
    """
    Split data using GroupShuffleSplit based on a grouping column
    (e.g., 'name' in Steel DB or 'alloy_code' in MatNavi).

    This strictly ensures that multiple tests/dimensions of the same alloy
    never leak between the training set and the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features, target, and grouping column.
    feature_cols : list of str
        List of input feature column names.
    target_col : str
        Target column name to predict.
    group_col : str
        Column defining independent material groups ('name' or 'alloy_code').
    test_size : float
        Fraction of groups reserved for testing (default 0.2).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test, groups_train, groups_test
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[group_col]

    train_idx, test_idx = next(gss.split(df, groups=groups))

    X_train = df.iloc[train_idx][feature_cols].copy()
    X_test = df.iloc[test_idx][feature_cols].copy()
    y_train = df.iloc[train_idx][target_col].copy()
    y_test = df.iloc[test_idx][target_col].copy()

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    # Verification: check for overlap of material groups
    overlap = set(groups_train.unique()).intersection(set(groups_test.unique()))
    if len(overlap) > 0:
        raise ValueError(f"CRITICAL ERROR: Data leakage detected! Overlapping groups: {overlap}")

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def create_strength_categories(df, target_col="tensile_strength_mpa"):
    """
    Assign a three-tier material strength category based on tensile strength:
      - 'Low Strength': < 600 MPa (mild, plain carbon, structural steels)
      - 'Medium Strength': 600 to 950 MPa (machinery, quenched & tempered steels)
      - 'High Strength': >= 950 MPa (high strength structural, spring, maraging steels)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing tensile strength.
    target_col : str
        Name of tensile strength column.

    Returns
    -------
    pd.Series
        Categorical labels for each row.
    """
    ts = df[target_col]
    conditions = [
        (ts < 600),
        ((ts >= 600) & (ts < 950)),
        (ts >= 950)
    ]
    choices = ["Low Strength", "Medium Strength", "High Strength"]
    categories = np.select(conditions, choices, default="Medium Strength")
    return pd.Series(categories, index=df.index, name="strength_category")


def evaluate_regression(y_true, y_pred):
    """
    Calculate regression evaluation metrics:
    MAE, MSE, RMSE, R² score.

    Returns
    -------
    dict
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 2),
        "MSE": round(float(mse), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4)
    }


def evaluate_classification(y_true, y_pred, y_prob=None, labels=None, prob_classes=None):
    """
    Calculate classification evaluation metrics:
    Accuracy, Macro Precision, Macro Recall, Macro F1-score, ROC-AUC, Confusion Matrix.

    Returns
    -------
    dict
    """
    if labels is None:
        labels = ["Low Strength", "Medium Strength", "High Strength"]
    if prob_classes is None:
        prob_classes = labels

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    roc_auc = None
    if y_prob is not None:
        try:
            # One-vs-Rest macro-averaged ROC-AUC matching estimator class probability columns
            roc_auc = round(float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=prob_classes)), 4)
        except Exception:
            roc_auc = None

    return {
        "Accuracy": round(float(acc), 4),
        "Precision_macro": round(float(prec), 4),
        "Recall_macro": round(float(rec), 4),
        "F1_macro": round(float(f1), 4),
        "ROC_AUC": roc_auc,
        "Confusion_Matrix": cm.tolist(),
        "Classification_Report": report
    }


def save_model_artifact(model, metadata, model_filename, subfolder="regression"):
    """
    Save trained model object and its metadata dictionary using joblib and json.

    Parameters
    ----------
    model : estimator or Pipeline
        Fitted Scikit-learn estimator/pipeline.
    metadata : dict
        Metadata including dataset, features, targets, metrics, hyperparams.
    model_filename : str
        e.g. 'linear_regression_tensile_steel_db.joblib'
    subfolder : str
        'regression', 'classification', 'clustering', or 'pca'
    """
    target_dir = os.path.join(MODELS_DIR, subfolder)
    os.makedirs(target_dir, exist_ok=True)

    model_path = os.path.join(target_dir, model_filename)
    joblib.dump(model, model_path)

    meta_filename = model_filename.replace(".joblib", "_meta.json")
    meta_path = os.path.join(target_dir, meta_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    return model_path, meta_path


def load_model_artifact(model_filename, subfolder="regression"):
    """
    Load a trained model and its metadata.

    Returns
    -------
    model, metadata
    """
    target_dir = os.path.join(MODELS_DIR, subfolder)
    model_path = os.path.join(target_dir, model_filename)
    meta_path = os.path.join(target_dir, model_filename.replace(".joblib", "_meta.json"))

    model = joblib.load(model_path)
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, metadata
