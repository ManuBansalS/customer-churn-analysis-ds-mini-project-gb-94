"""
Gold Layer - Model Training Module
====================================
Classification models for 3-class churn prediction.
Supports: Logistic Regression, Random Forest, Gradient Boosting, SVM.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def train_logistic_regression(X_train, y_train, random_state=42):
    """Train multiclass Logistic Regression with class balancing."""
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    print("Logistic Regression trained.")
    return model


def train_random_forest(X_train, y_train, random_state=42):
    """Train Random Forest with class balancing."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Random Forest trained.")
    return model


def train_gradient_boosting(X_train, y_train, random_state=42):
    """Train Gradient Boosting classifier."""
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("Gradient Boosting trained.")
    return model


def train_svm(X_train, y_train, random_state=42):
    """Train SVM with RBF kernel and probability estimates."""
    model = SVC(
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("SVM trained (RBF, OVR).")
    return model


def hyperparameter_tune_rf(X_train, y_train, random_state=42, cv=5):
    """GridSearchCV for Random Forest (optimise weighted F1)."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    rf = RandomForestClassifier(
        class_weight='balanced', random_state=random_state, n_jobs=-1
    )
    grid = GridSearchCV(
        rf, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    print(f"\nBest RF Params: {grid.best_params_}")
    print(f"Best RF F1-weighted (CV): {grid.best_score_:.4f}")
    return grid.best_estimator_, grid


def hyperparameter_tune_gb(X_train, y_train, random_state=42, cv=5):
    """GridSearchCV for Gradient Boosting (optimise weighted F1)."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
    }
    gb = GradientBoostingClassifier(random_state=random_state)
    grid = GridSearchCV(
        gb, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    print(f"\nBest GB Params: {grid.best_params_}")
    print(f"Best GB F1-weighted (CV): {grid.best_score_:.4f}")
    return grid.best_estimator_, grid
