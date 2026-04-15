"""
Gold Layer - Data Preprocessing Module
========================================
Prepares the customer-level dataset for 3-class churn modeling.

Target: no_churn (0), partial_churn (1), full_churn (2)
Features: Aggregated from BoB data (no data leakage).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_modeling_data(filepath):
    """
    Load the customer-level analysis dataset produced by the Silver layer.

    Parameters
    ----------
    filepath : str
        Path to analysis_data.csv.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath)
    print(f"Loaded modeling data: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def encode_target(df, target_col='churn_category'):
    """
    Encode the 3-class churn target as integers.
    no_churn=0, partial_churn=1, full_churn=2

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str

    Returns
    -------
    pd.DataFrame with 'churn_encoded' column, label mapping dict
    """
    df = df.copy()
    mapping = {'no_churn': 0, 'partial_churn': 1, 'full_churn': 2}
    df['churn_encoded'] = df[target_col].map(mapping)

    print("\n--- Target Distribution ---")
    for label, code in mapping.items():
        count = (df['churn_encoded'] == code).sum()
        pct = count / len(df) * 100
        print(f"  {code} ({label:>15s}): {count:>6d} ({pct:.1f}%)")

    return df, mapping


def get_feature_target_split(df, target_col='churn_encoded',
                              drop_cols=None):
    """
    Separate features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    drop_cols : list or None
        Additional columns to drop from features.

    Returns
    -------
    X, y, feature_names
    """
    if drop_cols is None:
        drop_cols = []

    # Always drop these non-feature columns
    always_drop = ['account_number', 'churn_category', 'churn_encoded',
                   'customer_lost_cases', 'customer_saved_cases',
                   'total_retention_cases', 'num_agreements_y',
                   'dominant_agreement_type']
    all_drop = list(set(always_drop + drop_cols))
    all_drop = [c for c in all_drop if c in df.columns]

    feature_cols = [c for c in df.columns if c not in all_drop and c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_cols


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """Stratified train/test split."""
    stratify_col = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=stratify_col
    )
    print(f"\n--- Train/Test Split ---")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    for label in sorted(y.unique()):
        tr_count = (y_train == label).sum()
        te_count = (y_test == label).sum()
        print(f"  Class {label}: train={tr_count}, test={te_count}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """StandardScaler fit on train, transform both."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled with StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler
