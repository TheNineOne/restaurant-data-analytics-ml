"""
Level 3 - Task 1: Predictive Modeling
=====================================
Internship: Cognifyz Technologies Data Science Internship
Project: Restaurant Data Analytics & Rating Prediction System

Tasks covered:
1. Build regression models to predict restaurant "Aggregate rating".
2. Split dataset into train and test sets (80/20 split, random_state=42).
3. Strictly prevent data leakage (exclude 'Rating color', 'Rating text', 'Restaurant ID').
4. Train and compare:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
5. Evaluate models using MAE, MSE, RMSE, and R2 (Train & Test).
6. Select the best-performing model based on test evaluation.
7. Save the trained best model using Joblib (models/best_model.joblib).
8. Save model metadata JSON (models/model_metadata.json).
9. Save comparative charts and evaluation report.
"""

import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cognifyz")
import json
import warnings
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


def find_data_file(filename="engineered_restaurant_data.csv"):
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", filename),
        os.path.join(os.path.dirname(__file__), "data", "processed", filename),
        os.path.join(os.getcwd(), "restaurant-data-science", "data", "processed", filename),
        os.path.join(os.getcwd(), "data", "processed", filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    # Fallback to raw dataset if engineered dataset not found
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Dataset .csv")
    if os.path.exists(raw_path):
        return os.path.abspath(raw_path)
    raise FileNotFoundError(f"Could not locate '{filename}' in candidate paths.")


def get_paths():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(base_dir, "models")
    charts_dir = os.path.join(base_dir, "outputs", "charts")
    reports_dir = os.path.join(base_dir, "outputs", "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return models_dir, charts_dir, reports_dir


def prepare_features(df):
    """Ensure all required features exist and are properly formatted."""
    data = df.copy()

    # Preprocessing
    if "Name_Length" not in data.columns:
        data["Name_Length"] = data["Restaurant Name"].astype(str).str.len()
    if "Address_Length" not in data.columns:
        data["Address_Length"] = data["Address"].astype(str).str.len()
    if "Has_Table_booking_encoded" not in data.columns:
        data["Has_Table_booking_encoded"] = data["Has Table booking"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    if "Has_Online_delivery_encoded" not in data.columns:
        data["Has_Online_delivery_encoded"] = data["Has Online delivery"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    if "Num_Cuisines" not in data.columns:
        data["Cuisines"] = data["Cuisines"].fillna("Unknown")
        data["Num_Cuisines"] = data["Cuisines"].apply(lambda x: len(str(x).split(", ")) if str(x) != "Unknown" else 0)

    feature_cols = [
        "Country Code",
        "Longitude",
        "Latitude",
        "Average Cost for two",
        "Price range",
        "Votes",
        "Name_Length",
        "Address_Length",
        "Has_Table_booking_encoded",
        "Has_Online_delivery_encoded",
        "Num_Cuisines"
    ]

    target_col = "Aggregate rating"

    X = data[feature_cols].copy()
    y = data[target_col].copy()

    # Impute any missing in features
    X = X.fillna(X.median())

    return X, y, feature_cols


def run_task_1(save_outputs=True):
    print("=" * 70)
    print("LEVEL 3 - TASK 1: PREDICTIVE MODELING")
    print("=" * 70)

    # 1. Load Data
    data_path = find_data_file("engineered_restaurant_data.csv")
    print(f"[INFO] Loading data from: {data_path}")
    df = pd.read_csv(data_path, encoding="utf-8")

    # 2. Prepare Features & Prevent Data Leakage
    # Explicitly verify NO Rating color or Rating text in features
    X, y, feature_cols = prepare_features(df)
    print(f"\n[1] Features Selected ({len(feature_cols)} features):")
    for f in feature_cols:
        print(f"    - {f}")
    print("\n[LEAKAGE CHECK] Excluded Columns: 'Rating color', 'Rating text', 'Restaurant ID'")
    print(f"                 Target Variable : 'Aggregate rating' (Shape: {y.shape})")

    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[2] Dataset Partitioning:")
    print(f"    - Train Set: {X_train.shape[0]:,} samples (80%)")
    print(f"    - Test Set : {X_test.shape[0]:,} samples (20%)")

    # 4. Model Definition
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
    }

    # 5. Model Training and Evaluation
    print("\n[3] Model Training and Evaluation on Test Set:")
    results = {}
    fitted_models = {}

    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Train Metrics
        train_mae = mean_absolute_error(y_train, train_preds)
        train_mse = mean_squared_error(y_train, train_preds)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, train_preds)

        # Test Metrics
        test_mae = mean_absolute_error(y_test, test_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, test_preds)

        # 5-Fold Cross Validation R2
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        cv_r2_mean = cv_scores.mean()

        results[name] = {
            "Train MAE": train_mae,
            "Train RMSE": train_rmse,
            "Train R2": train_r2,
            "Test MAE": test_mae,
            "Test MSE": test_mse,
            "Test RMSE": test_rmse,
            "Test R2": test_r2,
            "CV R2 (5-Fold)": cv_r2_mean,
            "Predictions": test_preds
        }

        print(f"    --- {name} ---")
        print(f"        Train: MAE={train_mae:.4f}, RMSE={train_rmse:.4f}, R2={train_r2:.4f}")
        print(f"        Test : MAE={test_mae:.4f}, MSE={test_mse:.4f}, RMSE={test_rmse:.4f}, R2={test_r2:.4f}")
        print(f"        5-Fold CV Mean R2: {cv_r2_mean:.4f}")

    # 6. Model Comparison Table
    comparison_df = pd.DataFrame({
        "Model": list(results.keys()),
        "MAE (Test)": [results[m]["Test MAE"] for m in results],
        "MSE (Test)": [results[m]["Test MSE"] for m in results],
        "RMSE (Test)": [results[m]["Test RMSE"] for m in results],
        "R2 Score (Test)": [results[m]["Test R2"] for m in results],
        "5-Fold CV R2": [results[m]["CV R2 (5-Fold)"] for m in results]
    }).sort_values(by="R2 Score (Test)", ascending=False)

    print("\n[4] Comprehensive Model Performance Benchmark:")
    print(comparison_df.to_string(index=False, justify="center"))

    # Best Model Selection
    best_model_name = comparison_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    best_r2 = results[best_model_name]["Test R2"]
    best_mae = results[best_model_name]["Test MAE"]
    best_rmse = results[best_model_name]["Test RMSE"]
    print(f"\n[5] Best-Performing Model Selected: '{best_model_name}'")
    print(f"    - Test R2   : {best_r2:.4f} ({best_r2*100:.2f}% variance explained)")
    print(f"    - Test MAE  : {best_mae:.4f} rating points")
    print(f"    - Test RMSE : {best_rmse:.4f} rating points")

    # Feature Importance for Tree Models
    rf_model = fitted_models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n[6] Random Forest Feature Importance Ranking:")
    for f_name, imp in importances.items():
        print(f"    - {f_name:<30}: {imp:.4f} ({imp*100:.2f}%)")

    # 7. Model Serialization & Outputs
    models_dir, charts_dir, reports_dir = get_paths()

    # Save Best Model with Joblib
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"\n[OUTPUT] Best model serialized to: {best_model_path}")

    # Save Model Metadata JSON
    metadata = {
        "model_name": best_model_name,
        "feature_names": feature_cols,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "evaluation_metrics": {
            "test_mae": float(best_mae),
            "test_mse": float(results[best_model_name]["Test MSE"]),
            "test_rmse": float(best_rmse),
            "test_r2": float(best_r2),
            "cv_r2_5fold": float(results[best_model_name]["CV R2 (5-Fold)"])
        },
        "all_models_comparison": {
            m: {
                "mae": float(results[m]["Test MAE"]),
                "mse": float(results[m]["Test MSE"]),
                "rmse": float(results[m]["Test RMSE"]),
                "r2": float(results[m]["Test R2"]),
                "cv_r2": float(results[m]["CV R2 (5-Fold)"])
            }
            for m in results
        },
        "feature_importances": {f: float(v) for f, v in importances.items()}
    }

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[OUTPUT] Model metadata saved to: {metadata_path}")

    # 8. Save Visualizations and Report
    if save_outputs:
        # Save Report
        report_file = os.path.join(reports_dir, "level_3_task_1_model_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("COGNIFYZ DATA SCIENCE INTERNSHIP - LEVEL 3 TASK 1 REPORT\n")
            f.write("PREDICTIVE MODELING & PERFORMANCE BENCHMARK\n")
            f.write("=" * 60 + "\n\n")
            f.write("1. DATA SPLIT:\n")
            f.write(f"   - Training Set: {len(X_train)} samples\n")
            f.write(f"   - Testing Set : {len(X_test)} samples\n\n")
            f.write("2. MODEL COMPARISON RESULTS:\n")
            f.write(comparison_df.to_string(index=False) + "\n\n")
            f.write(f"3. BEST MODEL: {best_model_name}\n")
            f.write(f"   - Test MAE : {best_mae:.4f}\n")
            f.write(f"   - Test RMSE: {best_rmse:.4f}\n")
            f.write(f"   - Test R2  : {best_r2:.4f}\n\n")
            f.write("4. RANDOM FOREST FEATURE IMPORTANCES:\n")
            for f_name, imp in importances.items():
                f.write(f"   - {f_name:<30}: {imp:.4f} ({imp*100:.2f}%)\n")
        print(f"[OUTPUT] Report saved to: {report_file}")

        # Visualizations
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # Chart 1: Model Benchmark Bar Charts
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # R2 Comparison
        sns.barplot(data=comparison_df, x="Model", y="R2 Score (Test)", ax=axes[0], palette=["#2a9d8f", "#e9c46a", "#e76f51"])
        axes[0].set_title("Model Comparison: Test R² Score (Higher is Better)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("R² Score", fontsize=11)
        axes[0].set_ylim(0, 1.1)
        for i, val in enumerate(comparison_df["R2 Score (Test)"]):
            axes[0].text(i, val + 0.03, f"{val:.4f}", ha="center", fontweight="bold", fontsize=10)

        # RMSE Comparison
        sns.barplot(data=comparison_df, x="Model", y="RMSE (Test)", ax=axes[1], palette=["#e76f51", "#e9c46a", "#2a9d8f"])
        axes[1].set_title("Model Comparison: Test RMSE (Lower is Better)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Root Mean Squared Error (RMSE)", fontsize=11)
        axes[1].set_ylim(0, 1.5)
        for i, val in enumerate(comparison_df["RMSE (Test)"]):
            axes[1].text(i, val + 0.03, f"{val:.4f}", ha="center", fontweight="bold", fontsize=10)

        plt.suptitle("Cognifyz Technologies - Level 3 Task 1: Machine Learning Model Benchmark", fontsize=14, fontweight="bold")
        plt.tight_layout()
        chart_file_1 = os.path.join(charts_dir, "level_3_task_1_model_comparison.png")
        fig.savefig(chart_file_1, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OUTPUT] Model comparison chart saved to: {chart_file_1}")

        # Chart 2: Actual vs Predicted & Feature Importances
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

        # Actual vs Predicted (Random Forest)
        rf_preds = results["Random Forest"]["Predictions"]
        axes2[0].scatter(y_test, rf_preds, alpha=0.3, color="#2b5c8f", edgecolors="none")
        axes2[0].plot([0, 5], [0, 5], color="red", linestyle="--", linewidth=2, label="Ideal Line (y = x)")
        axes2[0].set_title("Actual vs Predicted Ratings (Random Forest)", fontsize=12, fontweight="bold")
        axes2[0].set_xlabel("Actual Aggregate Rating", fontsize=11)
        axes2[0].set_ylabel("Predicted Aggregate Rating", fontsize=11)
        axes2[0].legend(frameon=True)

        # Feature Importances Horizontal Bar
        sns.barplot(x=importances.values, y=importances.index, ax=axes2[1], palette="crest_r")
        axes2[1].set_title("Random Forest Feature Importances", fontsize=12, fontweight="bold")
        axes2[1].set_xlabel("Relative Importance Score", fontsize=11)
        for i, val in enumerate(importances.values):
            axes2[1].text(val + 0.01, i, f"{val:.3f}", va="center", fontweight="bold", fontsize=9)

        plt.suptitle("Cognifyz Technologies - Level 3 Task 1: Prediction Quality & Feature Drivers", fontsize=14, fontweight="bold")
        plt.tight_layout()
        chart_file_2 = os.path.join(charts_dir, "level_3_task_1_actual_vs_predicted.png")
        fig2.savefig(chart_file_2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"[OUTPUT] Prediction quality chart saved to: {chart_file_2}")

    print("\n" + "=" * 70)
    print("LEVEL 3 - TASK 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return comparison_df


if __name__ == "__main__":
    run_task_1()
