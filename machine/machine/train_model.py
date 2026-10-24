"""
Train the Logistic Regression model for maternal satisfaction prediction.
Extracted from the original Google Colab notebook (Python_Project (2).ipynb).

Usage:
    python train_model.py

This script:
1. Loads the SPSS .sav data file
2. Performs descriptive analysis (prints to console)
3. Handles missing values
4. One-hot encodes categorical features
5. Scales numerical features
6. Trains a Logistic Regression model
7. Evaluates the model
8. Saves all artifacts (model, scaler, feature columns, etc.) to the models/ directory
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, brier_score_loss, RocCurveDisplay
)
from sklearn.inspection import permutation_importance
import joblib

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Delivery care satisfaction.sav')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    # --- Load the SPSS data file ---
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at: {DATA_PATH}")
        print("Please copy your 'Delivery care satisfaction.sav' file into the 'data/' directory.")
        print("You can download it from your Google Drive where the Colab notebook stored it.")
        return

    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_spss(DATA_PATH)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print()

    # --- Part 1: Descriptive Analysis ---
    print("=" * 60)
    print("PART 1: Descriptive Analysis")
    print("=" * 60)

    print("\nDataFrame Info:")
    df.info()

    print("\nDescriptive Statistics for Numerical Columns:")
    print(df.describe())

    # Frequency counts for categorical columns
    categorical_cols = df.select_dtypes(include='category').columns
    print("\nFrequency Counts for Categorical Columns:")
    for col in categorical_cols:
        print(f"\n--- {col} ---")
        print(df[col].value_counts())
        print(f"Missing values in {col}: {df[col].isnull().sum()}")

    # Variance for numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print("\nVariance for Numerical Columns:")
    print(df[numerical_cols].var())

    # --- Part 2: Feature Engineering and Data Management ---
    print("\n" + "=" * 60)
    print("PART 2: Feature Engineering")
    print("=" * 60)

    # Step 1: Handling Missing Values
    missing_cols = df.columns[df.isnull().any()].tolist()
    print("Columns with missing values:", missing_cols)

    missing_percentages = df[missing_cols].isnull().sum() * 100 / len(df)
    print("\nPercentage of missing values per column:")
    print(missing_percentages.sort_values(ascending=False))

    # Drop columns with high percentage of missing values
    columns_to_drop = ['q1092disth', 'wolkingdistancerecodeinhrs']
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
    print(f"\nColumns dropped: {columns_to_drop}")
    print("DataFrame shape after dropping columns:", df.shape)

    # Step 2: Feature Encoding (One-Hot Encoding)
    categorical_features_for_encoding = df.select_dtypes(include='category').columns.tolist()
    if 'satisfaction' in categorical_features_for_encoding:
        categorical_features_for_encoding.remove('satisfaction')
    print(f"\nCategorical features to be encoded: {len(categorical_features_for_encoding)}")

    df_encoded = pd.get_dummies(df, columns=categorical_features_for_encoding, drop_first=True)
    print(f"DataFrame shape after One-Hot Encoding: {df_encoded.shape}")

    # Step 3: Feature Scaling preparation
    numerical_cols_for_scaling = df_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'satisfaction' in numerical_cols_for_scaling:
        numerical_cols_for_scaling.remove('satisfaction')
    print(f"\nNumerical features for scaling: {numerical_cols_for_scaling}")

    df_scaled = df_encoded.copy()

    # --- Part 3: Model Training ---
    print("\n" + "=" * 60)
    print("PART 3: Model Training")
    print("=" * 60)

    # Separate features (X) and target (y)
    X = df_scaled.drop('satisfaction', axis=1)
    y = df_scaled['satisfaction']

    # Convert target to numerical (0 = not satisfied, 1 = satisfied)
    y = y.map({'not satisfied': 0, 'satisfied': 1})

    # Train-test split (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train value counts:\n{y_train.value_counts()}")
    print(f"y_test value counts:\n{y_test.value_counts()}")

    # Fit scaler ONLY on training data, then transform both sets
    scaler = StandardScaler()
    X_train[numerical_cols_for_scaling] = scaler.fit_transform(X_train[numerical_cols_for_scaling])
    X_test[numerical_cols_for_scaling] = scaler.transform(X_test[numerical_cols_for_scaling])

    # Train Logistic Regression model
    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train, y_train)
    print("\nLogistic Regression model trained successfully.")

    # --- Part 4: Model Evaluation ---
    print("\n" + "=" * 60)
    print("PART 4: Model Evaluation")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Brier:     {brier:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")

    # --- Part 5: Feature Importance ---
    print("\n" + "=" * 60)
    print("PART 5: Feature Importance")
    print("=" * 60)

    # Coefficient-based importance
    feature_names = X_train.columns
    coefficients = model.coef_[0]
    feature_importance_lr = pd.Series(coefficients, index=feature_names)
    sorted_feature_importance_lr = feature_importance_lr.reindex(
        feature_importance_lr.abs().sort_values(ascending=False).index
    )
    print("\nTop 20 Features by Absolute Coefficient Value:")
    print(sorted_feature_importance_lr.head(20))

    # Permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    feature_importances_perm = pd.Series(perm_importance.importances_mean, index=X_test.columns)
    sorted_feature_importance_perm = feature_importances_perm.sort_values(ascending=False)
    print("\nTop 20 Features by Permutation Importance:")
    print(sorted_feature_importance_perm.head(20))

    # --- Part 5.1: Recursive Feature Elimination (RFE) ---
    print("\n" + "=" * 60)
    print("PART 5.1: Recursive Feature Elimination (RFE)")
    print("=" * 60)

    estimator = LogisticRegression(random_state=42, solver='liblinear')
    n_features_to_select = 10
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector = selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.support_]
    print(f"Top {n_features_to_select} Selected Features:")
    for idx, feature in enumerate(selected_features, 1):
        print(f"  {idx}. {feature}")

    # --- Part 6: Save Model Artifacts ---
    print("\n" + "=" * 60)
    print("PART 6: Saving Model Artifacts")
    print("=" * 60)

    joblib.dump(model, os.path.join(MODELS_DIR, 'logistic_regression_model.joblib'))
    print("Saved: logistic_regression_model.joblib")

    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    print("Saved: scaler.joblib")

    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, 'feature_columns.joblib'))
    print("Saved: feature_columns.joblib")

    joblib.dump(categorical_features_for_encoding, os.path.join(MODELS_DIR, 'categorical_features_encoded.joblib'))
    print("Saved: categorical_features_encoded.joblib")

    joblib.dump(y_test, os.path.join(MODELS_DIR, 'y_test.joblib'))
    print("Saved: y_test.joblib")

    joblib.dump(y_pred, os.path.join(MODELS_DIR, 'y_pred.joblib'))
    print("Saved: y_pred.joblib")

    joblib.dump(sorted_feature_importance_lr, os.path.join(MODELS_DIR, 'coef_importance.joblib'))
    print("Saved: coef_importance.joblib")

    joblib.dump(sorted_feature_importance_perm, os.path.join(MODELS_DIR, 'perm_importance.joblib'))
    print("Saved: perm_importance.joblib")

    print("\n" + "=" * 60)
    print("Training complete! All model artifacts saved to 'models/' directory.")
    print("You can now run the Streamlit app with: streamlit run app.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
