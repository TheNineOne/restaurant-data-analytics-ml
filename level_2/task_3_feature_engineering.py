"""
Level 2 - Task 3: Feature Engineering
=====================================
Internship: Cognifyz Technologies Data Science Internship
Project: Restaurant Data Analytics & Rating Prediction System

Tasks covered:
1. Extract length of restaurant name ('Name_Length').
2. Extract length of restaurant address ('Address_Length').
3. Create 'Has Table Booking' encoded binary feature (1 / 0).
4. Create 'Has Online Delivery' encoded binary feature (1 / 0).
5. Extract additional high-value engineered features (Num_Cuisines, Country_Name, Is_Delivering_now_encoded).
6. Save processed and engineered datasets separately:
   - data/processed/cleaned_restaurant_data.csv
   - data/processed/engineered_restaurant_data.csv
7. Save feature engineering report and correlation visualization.
"""

import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cognifyz")
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

COUNTRY_CODE_MAP = {
    1: "India",
    14: "Australia",
    30: "Brazil",
    37: "Canada",
    94: "Indonesia",
    148: "New Zealand",
    162: "Philippines",
    166: "Qatar",
    184: "Singapore",
    189: "South Africa",
    191: "Sri Lanka",
    208: "Turkey",
    214: "UAE",
    215: "United Kingdom",
    216: "United States"
}


def find_data_file(filename="Dataset .csv"):
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", filename),
        os.path.join(os.path.dirname(__file__), "data", "raw", filename),
        os.path.join(os.getcwd(), "restaurant-data-science", "data", "raw", filename),
        os.path.join(os.getcwd(), "data", "raw", filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"Could not locate '{filename}' in candidate paths.")


def get_paths():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    processed_dir = os.path.join(base_dir, "data", "processed")
    charts_dir = os.path.join(base_dir, "outputs", "charts")
    reports_dir = os.path.join(base_dir, "outputs", "reports")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return processed_dir, charts_dir, reports_dir


def run_task_3(save_outputs=True):
    print("=" * 70)
    print("LEVEL 2 - TASK 3: FEATURE ENGINEERING")
    print("=" * 70)

    # 1. Load Data
    data_path = find_data_file("Dataset .csv")
    df = pd.read_csv(data_path, encoding="utf-8")
    initial_shape = df.shape
    print(f"[1] Original Dataset Loaded: {initial_shape[0]:,} rows x {initial_shape[1]} columns")

    # 2. Impute and Clean Missing Values
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    # 3. Feature Engineering: Name and Address Length
    print("\n[2] Extracting String Length Features:")
    df["Name_Length"] = df["Restaurant Name"].astype(str).str.len()
    df["Address_Length"] = df["Address"].astype(str).str.len()
    print(f"    - 'Name_Length'    : Min={df['Name_Length'].min()}, Max={df['Name_Length'].max()}, Mean={df['Name_Length'].mean():.2f}")
    print(f"    - 'Address_Length' : Min={df['Address_Length'].min()}, Max={df['Address_Length'].max()}, Mean={df['Address_Length'].mean():.2f}")

    # 4. Categorical Binary Encoding
    print("\n[3] Encoding Binary Categorical Features (Yes -> 1, No -> 0):")
    df["Has_Table_booking_encoded"] = df["Has Table booking"].map({"Yes": 1, "No": 0}).astype(int)
    df["Has_Online_delivery_encoded"] = df["Has Online delivery"].map({"Yes": 1, "No": 0}).astype(int)
    df["Is_Delivering_now_encoded"] = df["Is delivering now"].map({"Yes": 1, "No": 0}).astype(int)
    print(f"    - 'Has_Table_booking_encoded'   : {df['Has_Table_booking_encoded'].sum():,} positives (1s)")
    print(f"    - 'Has_Online_delivery_encoded' : {df['Has_Online_delivery_encoded'].sum():,} positives (1s)")
    print(f"    - 'Is_Delivering_now_encoded'   : {df['Is_Delivering_now_encoded'].sum():,} positives (1s)")

    # 5. Additional Domain Features
    print("\n[4] Creating Domain-Specific Features:")
    df["Num_Cuisines"] = df["Cuisines"].apply(lambda x: len(x.split(", ")) if x != "Unknown" else 0)
    df["Country_Name"] = df["Country Code"].map(COUNTRY_CODE_MAP).fillna("Other")
    price_tier_map = {1: "Budget", 2: "Mid-Range", 3: "Expensive", 4: "Luxury"}
    df["Price_Category"] = df["Price range"].map(price_tier_map)
    print(f"    - 'Num_Cuisines' : Min={df['Num_Cuisines'].min()}, Max={df['Num_Cuisines'].max()}, Mean={df['Num_Cuisines'].mean():.2f}")
    print(f"    - 'Country_Name' : Mapped {df['Country_Name'].nunique()} country names")

    # 6. Feature Correlation with Target Variable
    engineered_cols = [
        "Aggregate rating",
        "Votes",
        "Price range",
        "Average Cost for two",
        "Name_Length",
        "Address_Length",
        "Has_Table_booking_encoded",
        "Has_Online_delivery_encoded",
        "Num_Cuisines"
    ]
    corr_matrix = df[engineered_cols].corr()
    print("\n[5] Pearson Correlation with 'Aggregate rating':")
    target_corrs = corr_matrix["Aggregate rating"].sort_values(ascending=False)
    for col, val in target_corrs.items():
        if col != "Aggregate rating":
            print(f"    - {col:<30}: {val:+.4f}")

    # 7. Save Cleaned & Engineered Datasets
    processed_dir, charts_dir, reports_dir = get_paths()

    # (A) Cleaned dataset
    cleaned_file = os.path.join(processed_dir, "cleaned_restaurant_data.csv")
    cleaned_df = df.copy()
    cleaned_df.to_csv(cleaned_file, index=False, encoding="utf-8")
    print(f"\n[OUTPUT] Saved Cleaned Dataset to: {cleaned_file}")

    # (B) Engineered dataset
    engineered_file = os.path.join(processed_dir, "engineered_restaurant_data.csv")
    df.to_csv(engineered_file, index=False, encoding="utf-8")
    print(f"[OUTPUT] Saved Engineered Dataset to: {engineered_file}")
    print(f"         Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns (+{df.shape[1] - initial_shape[1]} engineered features)")

    # 8. Save Outputs
    if save_outputs:
        # Save Report
        report_file = os.path.join(reports_dir, "level_2_task_3_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("COGNIFYZ DATA SCIENCE INTERNSHIP - LEVEL 2 TASK 3 REPORT\n")
            f.write("FEATURE ENGINEERING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"1. Dataset Evolution:\n")
            f.write(f"   - Initial Columns: {initial_shape[1]}\n")
            f.write(f"   - Engineered Columns: {df.shape[1]}\n")
            f.write(f"   - New Features Added:\n")
            f.write("     * Name_Length (length of restaurant name)\n")
            f.write("     * Address_Length (length of address)\n")
            f.write("     * Has_Table_booking_encoded (1 for Yes, 0 for No)\n")
            f.write("     * Has_Online_delivery_encoded (1 for Yes, 0 for No)\n")
            f.write("     * Is_Delivering_now_encoded (1 for Yes, 0 for No)\n")
            f.write("     * Num_Cuisines (count of cuisines offered)\n")
            f.write("     * Country_Name (human-readable country name)\n")
            f.write("     * Price_Category (Budget, Mid-Range, Expensive, Luxury)\n\n")
            f.write("2. Correlation with Target ('Aggregate rating'):\n")
            for col, val in target_corrs.items():
                f.write(f"   - {col:<30}: {val:+.4f}\n")
        print(f"[OUTPUT] Report saved to: {report_file}")

        # Visualizations
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Chart 1: Heatmap of Engineered Features
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            vmin=-0.5,
            vmax=1.0,
            ax=axes[0],
            cbar_kws={"shrink": 0.8}
        )
        axes[0].set_title("Correlation Heatmap: Engineered Features & Rating", fontsize=12, fontweight="bold")

        # Chart 2: Name Length and Address Length vs Rating
        sns.regplot(
            data=df.sample(1000, random_state=42),
            x="Address_Length",
            y="Aggregate rating",
            ax=axes[1],
            scatter_kws={"alpha": 0.3, "color": "#2a9d8f"},
            line_kws={"color": "#e76f51", "linewidth": 2}
        )
        axes[1].set_title("Sample Scatter: Address Length vs Aggregate Rating", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Address Length (characters)", fontsize=11)
        axes[1].set_ylabel("Aggregate Rating", fontsize=11)

        plt.suptitle("Cognifyz Technologies - Level 2 Task 3: Feature Engineering Analysis", fontsize=15, fontweight="bold")
        plt.tight_layout()
        chart_file = os.path.join(charts_dir, "level_2_task_3_engineered_features.png")
        fig.savefig(chart_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OUTPUT] Chart saved to: {chart_file}")

    print("\n" + "=" * 70)
    print("LEVEL 2 - TASK 3 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return df


if __name__ == "__main__":
    run_task_3()
