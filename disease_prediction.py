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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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


def save_model_comparison_plot(model_names: list[str], accuracies: list[float], output_path: str) -> None:
    """Plot and save model accuracy comparison chart."""
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(model_names, accuracies, color=colors)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    # Label each bar with accuracy value for easier presentation.
    for bar, score in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_curve_plot(
    y_true: pd.Series,
    model_probabilities: dict[str, np.ndarray],
    output_path: str,
) -> dict[str, float]:
    """Plot and save ROC curves for all models, and return AUC values."""
    plt.figure(figsize=(9, 6))
    auc_scores: dict[str, float] = {}

    for model_name, y_prob in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        auc_scores[model_name] = roc_auc
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return auc_scores


def save_age_vs_target_boxplot(data: pd.DataFrame, output_path: str) -> None:
    """Plot and save age distribution grouped by target."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="target", y="age", data=data, palette="Set3", hue="target", legend=False)
    plt.title("Age vs Heart Disease (Target)")
    plt.xlabel("Target (0 = No Disease, 1 = Disease)")
    plt.ylabel("Age")
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
    save_age_vs_target_boxplot(df, "age_vs_target_boxplot.png")

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
    # 5A. MODEL COMPARISON SECTION
    # =============================
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5)

    logistic_model.fit(X_train_scaled, y_train)
    knn_model.fit(X_train_scaled, y_train)

    logistic_pred = logistic_model.predict(X_test_scaled)
    knn_pred = knn_model.predict(X_test_scaled)

    logistic_accuracy = accuracy_score(y_test, logistic_pred)
    rf_accuracy = accuracy
    knn_accuracy = accuracy_score(y_test, knn_pred)

    print("\nModel Accuracy Comparison (Side by Side):")
    print(f"- Logistic Regression: {logistic_accuracy:.4f}")
    print(f"- Random Forest:       {rf_accuracy:.4f}")
    print(f"- K-Nearest Neighbors: {knn_accuracy:.4f}")

    model_names = ["Logistic Regression", "Random Forest", "K-Nearest Neighbors"]
    model_accuracies = [logistic_accuracy, rf_accuracy, knn_accuracy]
    save_model_comparison_plot(model_names, model_accuracies, "model_comparison.png")

    # =============================
    # 5B. ROC CURVE SECTION
    # =============================
    logistic_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]
    rf_prob = model.predict_proba(X_test_scaled)[:, 1]
    knn_prob = knn_model.predict_proba(X_test_scaled)[:, 1]

    auc_scores = save_roc_curve_plot(
        y_test,
        {
            "Logistic Regression": logistic_prob,
            "Random Forest": rf_prob,
            "K-Nearest Neighbors": knn_prob,
        },
        "roc_curve.png",
    )

    # =============================
    # 6. CONCLUSION
    # =============================
    feature_importance_series = pd.Series(model.feature_importances_, index=X.columns)
    top_3_features = feature_importance_series.sort_values(ascending=False).head(3)

    model_summary = pd.DataFrame(
        {
            "Model": model_names,
            "Accuracy": model_accuracies,
            "AUC": [
                auc_scores["Logistic Regression"],
                auc_scores["Random Forest"],
                auc_scores["K-Nearest Neighbors"],
            ],
        }
    ).sort_values("Accuracy", ascending=False)

    print("\nModel Comparison Table (Model | Accuracy | AUC):")
    print(model_summary.to_string(index=False, formatters={"Accuracy": "{:.4f}".format, "AUC": "{:.4f}".format}))

    best_model_name = model_summary.iloc[0]["Model"]
    best_model_accuracy = model_summary.iloc[0]["Accuracy"]
    print(f"\nBest Performing Model: {best_model_name} (Accuracy: {best_model_accuracy:.4f})")

    print("\nTop 3 Most Important Features:")
    for feature, importance in top_3_features.items():
        print(f"- {feature}: {importance:.4f}")

    print(
        f"\nFinal Summary: The Random Forest model achieved {accuracy:.2%} accuracy "
        "in predicting heart disease on the test dataset."
    )


if __name__ == "__main__":
    main()
