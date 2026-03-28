"""
Disease Prediction System - Heart Disease UCI Dataset
Data Mining Course Project
"""

# =============================
# 1. IMPORTS
# =============================
import os

import matplotlib

# Use a non-interactive backend so plots can be saved without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Set a clean visual style for all plots.
sns.set_theme(style="whitegrid")


def save_target_count_plot(data: pd.DataFrame, output_path: str) -> None:
    """Plot and save target class distribution."""
    plt.figure(figsize=(8, 5))
    sns.countplot(x="target", data=data, palette="Set2", hue="target", legend=False)
    plt.title("Heart Disease Target Distribution")
    plt.xlabel("Target (0 = No Disease, 1 = Disease)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_age_distribution_plot(data: pd.DataFrame, output_path: str) -> None:
    """Plot and save age distribution."""
    plt.figure(figsize=(8, 5))
    sns.histplot(data["age"], kde=True, bins=20, color="steelblue")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_correlation_heatmap(data: pd.DataFrame, output_path: str) -> None:
    """Plot and save correlation heatmap."""
    plt.figure(figsize=(12, 9))
    corr_matrix = data.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_confusion_matrix_plot(y_true: pd.Series, y_pred: np.ndarray, output_path: str) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_feature_importance_plot(feature_names: pd.Index, importances: np.ndarray, output_path: str) -> None:
    """Plot and save feature importance chart."""
    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x="Importance", y="Feature", color="teal")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    # =============================
    # 2. DATA LOADING & EDA
    # =============================
    data_path = "heart.csv"

    if not os.path.exists(data_path):
        print(f"Error: '{data_path}' not found in current directory: {os.getcwd()}")
        print("Please place heart.csv in this folder and run again.")
        return

    df = pd.read_csv(data_path)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nDataset Info:")
    df.info()

    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nNull Values in Each Column:")
    print(df.isnull().sum())

    # Save EDA plots for presentation use.
    save_target_count_plot(df, "target_value_count.png")
    save_age_distribution_plot(df, "age_distribution.png")
    save_correlation_heatmap(df, "correlation_heatmap.png")

    # =============================
    # 3. DATA PREPROCESSING
    # =============================
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_rows = initial_rows - len(df)
    print(f"\nDuplicates removed: {removed_rows}")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =============================
    # 4. MODEL TRAINING
    # =============================
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # =============================
    # 5. RESULT ANALYSIS
    # =============================
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy Score:")
    print(f"{accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    save_confusion_matrix_plot(y_test, y_pred, "confusion_matrix.png")
    save_feature_importance_plot(X.columns, model.feature_importances_, "feature_importance.png")

    # =============================
    # 6. CONCLUSION
    # =============================
    feature_importance_series = pd.Series(model.feature_importances_, index=X.columns)
    top_3_features = feature_importance_series.sort_values(ascending=False).head(3)

    print("Top 3 Most Important Features:")
    for feature, importance in top_3_features.items():
        print(f"- {feature}: {importance:.4f}")

    print(
        f"\nFinal Summary: The Random Forest model achieved {accuracy:.2%} accuracy "
        "in predicting heart disease on the test dataset."
    )


if __name__ == "__main__":
    main()
