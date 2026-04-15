"""
Gold Layer - Model Evaluation Module
======================================
Evaluation functions for 3-class churn classification.
Supports multiclass confusion matrix, per-class ROC, feature importance,
model comparison, and model serialization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib

CLASS_NAMES = ['No Churn', 'Partial Churn', 'Full Churn']


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate multiclass model and return key metrics.

    Returns dict with Accuracy, Precision (weighted), Recall (weighted),
    F1 (weighted), and per-class metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }

    present_classes = sorted(y_test.unique())
    present_names = [CLASS_NAMES[c] for c in present_classes]

    print(f"\n{'=' * 60}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        if k != 'Model':
            print(f"  {k:>12s}: {v:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=present_names, zero_division=0)}")

    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name='Model'):
    """Plot multiclass confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    present_classes = sorted(y_test.unique())
    present_names = [CLASS_NAMES[c] for c in present_classes]

    cm = confusion_matrix(y_test, y_pred, labels=present_classes)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_names, yticklabels=present_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_roc_curves_multiclass(model, X_test, y_test, model_name='Model'):
    """Plot One-vs-Rest ROC curves for each class."""
    present_classes = sorted(y_test.unique())
    n_classes = len(present_classes)
    y_test_bin = label_binarize(y_test, classes=present_classes)

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    else:
        print(f"  {model_name}: predict_proba not available. Skipping ROC.")
        return

    # Ensure y_proba columns align with present_classes
    if y_proba.shape[1] != n_classes:
        print(f"  Warning: probability shape mismatch. Skipping ROC.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for i, cls in enumerate(present_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[cls % len(colors)], lw=2,
                label=f'{CLASS_NAMES[cls]} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves (OvR) - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, model_name='Model', top_n=15):
    """Plot feature importances for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        print(f"  {model_name}: no feature_importances_. Skipping.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(range(len(indices)), importances[indices][::-1],
            color=colors[::-1], edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=11)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def compare_models(results_list):
    """Side-by-side bar chart comparing metrics across models."""
    df = pd.DataFrame(results_list)
    df.set_index('Model', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

    return df


def save_model(model, filepath):
    """Save model with joblib."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load model with joblib."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
