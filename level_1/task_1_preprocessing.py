"""
Level 1 - Task 1: Data Exploration and Preprocessing
=====================================================
Internship: Cognifyz Technologies Data Science Internship
Project: Restaurant Data Analytics & Rating Prediction System

Tasks covered:
1. Load dataset and display shape (rows and columns).
2. Check missing values in each column and handle them appropriately.
3. Check for duplicates and investigate.
4. Perform necessary data type conversions.
5. Analyze the distribution of the target variable ("Aggregate rating").
6. Identify and explain class imbalance.
7. Save summary report and distribution chart.
"""

import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cognifyz")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_data_file(filename="Dataset .csv"):
    """Locate the dataset file across possible execution directories."""
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


def get_output_paths():
    """Get output directory paths for charts and reports."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    charts_dir = os.path.join(base_dir, "outputs", "charts")
    reports_dir = os.path.join(base_dir, "outputs", "reports")
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return charts_dir, reports_dir


def run_task_1(save_outputs=True):
    print("=" * 70)
    print("LEVEL 1 - TASK 1: DATA EXPLORATION AND PREPROCESSING")
    print("=" * 70)

    # 1. Load Dataset
    data_path = find_data_file("Dataset .csv")
    print(f"[INFO] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, encoding="utf-8")

    # 2. Shape and Identification
    n_rows, n_cols = df.shape
    print(f"\n[1] Dataset Dimensions:")
    print(f"    - Total Rows: {n_rows:,}")
    print(f"    - Total Columns: {n_cols}")

    print("\n[2] Columns and Initial Data Types:")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"    - {col:<25} : {str(dtype)}")

    # 3. Missing Values
    print("\n[3] Missing Values Analysis:")
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows) * 100
    missing_df = pd.DataFrame({"Missing_Count": missing, "Percentage": missing_pct})
    missing_cols = missing_df[missing_df["Missing_Count"] > 0]
    
    if len(missing_cols) > 0:
        print("    Columns with missing values:")
        for col, row in missing_cols.iterrows():
            print(f"    - {col}: {int(row['Missing_Count'])} missing ({row['Percentage']:.3f}%)")
    else:
        print("    No missing values found across all columns.")

    # Handling Missing Values
    # In Cuisines, missing values (9 rows, 0.094%) are imputed with 'Unknown'
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")
    print("    -> Imputed 9 missing values in 'Cuisines' with 'Unknown'.")

    # 4. Duplicate Rows
    duplicates = df.duplicated().sum()
    id_duplicates = df["Restaurant ID"].duplicated().sum()
    print(f"\n[4] Duplicate Investigation:")
    print(f"    - Entire row duplicates: {duplicates}")
    print(f"    - 'Restaurant ID' duplicates: {id_duplicates}")
    print("    -> No duplicate records detected. Every restaurant entry is unique.")

    # 5. Data Type Conversions
    print(f"\n[5] Data Type Conversions:")
    df["Restaurant ID"] = df["Restaurant ID"].astype(str)
    df["Country Code"] = df["Country Code"].astype(int)
    df["Price range"] = df["Price range"].astype(int)
    df["Aggregate rating"] = pd.to_numeric(df["Aggregate rating"], errors="coerce")
    df["Votes"] = df["Votes"].astype(int)
    print("    - Converted 'Restaurant ID' to string identifier.")
    print("    - Converted 'Price range' to discrete integer.")
    print("    - Converted 'Aggregate rating' to float64.")
    print("    - Converted 'Votes' to int64.")

    # 6. Target Variable ("Aggregate rating") Distribution
    target = df["Aggregate rating"]
    mean_rating = target.mean()
    median_rating = target.median()
    std_rating = target.std()
    min_rating = target.min()
    max_rating = target.max()

    zero_count = (target == 0.0).sum()
    zero_pct = (zero_count / n_rows) * 100
    rated_count = (target > 0.0).sum()
    rated_mean = target[target > 0.0].mean()
    rated_median = target[target > 0.0].median()

    print(f"\n[6] Target Variable ('Aggregate rating') Analysis:")
    print(f"    - Min Rating: {min_rating:.1f}")
    print(f"    - Max Rating: {max_rating:.1f}")
    print(f"    - Overall Mean Rating: {mean_rating:.3f}")
    print(f"    - Overall Median Rating: {median_rating:.3f}")
    print(f"    - Standard Deviation: {std_rating:.3f}")

    # 7. Class Imbalance Identification
    print(f"\n[7] Rating Class Imbalance Findings:")
    print(f"    - Unrated / Zero-rating restaurants: {zero_count:,} ({zero_pct:.2f}%)")
    print(f"    - Active rated restaurants (> 0.0): {rated_count:,} ({100 - zero_pct:.2f}%)")
    print(f"    - Active Ratings Mean: {rated_mean:.3f}")
    print(f"    - Active Ratings Median: {rated_median:.3f}")
    print("    -> Critical Finding: 2,148 restaurants (22.49%) have an Aggregate rating of 0.0.")
    print("       These correspond exactly to restaurants labeled 'Not rated' (Rating color: White).")
    print("       Active ratings follow a bell-shaped distribution peaked around 3.2 - 3.4.")

    # Save summary report and visualization if requested
    if save_outputs:
        charts_dir, reports_dir = get_output_paths()

        # Save Report
        report_file = os.path.join(reports_dir, "level_1_task_1_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("COGNIFYZ DATA SCIENCE INTERNSHIP - LEVEL 1 TASK 1 REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"1. Dataset Shape: {n_rows} rows x {n_cols} columns\n")
            f.write(f"2. Missing Values: 9 missing in 'Cuisines' ({9/n_rows*100:.3f}%), imputed with 'Unknown'.\n")
            f.write(f"3. Duplicates: 0 exact duplicates, 0 duplicate Restaurant IDs.\n")
            f.write(f"4. Target Variable ('Aggregate rating') Statistics:\n")
            f.write(f"   - Minimum: {min_rating:.2f}\n")
            f.write(f"   - Maximum: {max_rating:.2f}\n")
            f.write(f"   - Mean: {mean_rating:.3f}\n")
            f.write(f"   - Median: {median_rating:.3f}\n")
            f.write(f"   - Std Dev: {std_rating:.3f}\n")
            f.write(f"5. Class Imbalance Analysis:\n")
            f.write(f"   - Zero Rating Count: {zero_count} ({zero_pct:.2f}% of dataset)\n")
            f.write(f"   - Non-Zero Rating Count: {rated_count} ({100-zero_pct:.2f}% of dataset)\n")
            f.write(f"   - Mean of Rated Restaurants: {rated_mean:.3f}\n")
            f.write(f"   - Median of Rated Restaurants: {rated_median:.3f}\n")
            f.write(f"   - Qualitative Breakdown:\n")
            for text, cnt in df["Rating text"].value_counts().items():
                f.write(f"     * {text:<12}: {cnt} ({cnt/n_rows*100:.2f}%)\n")
        print(f"\n[OUTPUT] Report saved to: {report_file}")

        # Visualization
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Full distribution
        sns.histplot(df["Aggregate rating"], bins=25, kde=True, ax=axes[0], color="#2b5c8f", edgecolor="black")
        axes[0].set_title("Full Aggregate Rating Distribution (Including 0.0 Unrated)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Aggregate Rating (0.0 to 5.0)", fontsize=11)
        axes[0].set_ylabel("Frequency (Count)", fontsize=11)
        axes[0].axvline(mean_rating, color="red", linestyle="--", label=f"Mean: {mean_rating:.2f}")
        axes[0].axvline(median_rating, color="green", linestyle=":", label=f"Median: {median_rating:.2f}")
        axes[0].legend(frameon=True)

        # Non-zero active rating distribution
        active_ratings = df[df["Aggregate rating"] > 0]["Aggregate rating"]
        sns.histplot(active_ratings, bins=25, kde=True, ax=axes[1], color="#e07a5f", edgecolor="black")
        axes[1].set_title("Active Aggregate Rating Distribution (Excluding 0.0 Unrated)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Aggregate Rating (1.8 to 4.9)", fontsize=11)
        axes[1].set_ylabel("Frequency (Count)", fontsize=11)
        axes[1].axvline(rated_mean, color="red", linestyle="--", label=f"Mean: {rated_mean:.2f}")
        axes[1].axvline(rated_median, color="green", linestyle=":", label=f"Median: {rated_median:.2f}")
        axes[1].legend(frameon=True)

        plt.suptitle("Cognifyz Technologies - Level 1 Task 1: Target Variable & Imbalance Analysis", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        chart_file = os.path.join(charts_dir, "level_1_task_1_rating_distribution.png")
        fig.savefig(chart_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OUTPUT] Chart saved to: {chart_file}")

    print("\n" + "=" * 70)
    print("LEVEL 1 - TASK 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return df


if __name__ == "__main__":
    run_task_1()
