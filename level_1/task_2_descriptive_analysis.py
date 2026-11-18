"""
Level 1 - Task 2: Descriptive Analysis
=======================================
Internship: Cognifyz Technologies Data Science Internship
Project: Restaurant Data Analytics & Rating Prediction System

Tasks covered:
1. Calculate statistical measures (mean, median, standard deviation, min, max, IQR) for numerical columns.
2. Explore categorical distributions: Country Code, City, Cuisines.
3. Identify top 10 cuisines by restaurant count.
4. Identify top 10 cities with highest number of restaurants.
5. Save comprehensive descriptive report and visualizations.
"""

import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cognifyz")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def get_output_paths():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    charts_dir = os.path.join(base_dir, "outputs", "charts")
    reports_dir = os.path.join(base_dir, "outputs", "reports")
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return charts_dir, reports_dir


def run_task_2(save_outputs=True):
    print("=" * 70)
    print("LEVEL 1 - TASK 2: DESCRIPTIVE ANALYSIS")
    print("=" * 70)

    # 1. Load Dataset
    data_path = find_data_file("Dataset .csv")
    df = pd.read_csv(data_path, encoding="utf-8")
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")
    df["Country_Name"] = df["Country Code"].map(COUNTRY_CODE_MAP).fillna("Other")

    # 2. Numerical Summary Statistics
    num_cols = ["Average Cost for two", "Price range", "Aggregate rating", "Votes"]
    print("\n[1] Descriptive Statistics for Numerical Variables:")
    stats_list = []
    for col in num_cols:
        series = df[col]
        mean_val = series.mean()
        std_val = series.std()
        median_val = series.median()
        min_val = series.min()
        max_val = series.max()
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr_val = q75 - q25
        skew_val = series.skew()
        
        stats_list.append({
            "Column": col,
            "Mean": mean_val,
            "Std Dev": std_val,
            "Median": median_val,
            "Min": min_val,
            "25%": q25,
            "75%": q75,
            "IQR": iqr_val,
            "Max": max_val,
            "Skewness": skew_val
        })
    stats_df = pd.DataFrame(stats_list)
    print(stats_df.to_string(index=False, justify="center"))

    # 3. Categorical Variables: Country, City, Cuisines
    print("\n[2] Distribution of Categorical Variables:")
    print(f"    - Unique Countries: {df['Country Code'].nunique()}")
    print(f"    - Unique Cities: {df['City'].nunique()}")
    print(f"    - Unique Cuisines Combinations: {df['Cuisines'].nunique()}")

    print("\n[3] Restaurant Distribution by Country:")
    country_counts = df["Country_Name"].value_counts()
    for country, count in country_counts.items():
        pct = (count / len(df)) * 100
        print(f"    - {country:<22} : {count:>5} ({pct:>6.2f}%)")

    # 4. Top 10 Cities
    print("\n[4] Top 10 Cities with Highest Number of Restaurants:")
    top_10_cities = df["City"].value_counts().head(10)
    for rank, (city, count) in enumerate(top_10_cities.items(), 1):
        pct = (count / len(df)) * 100
        print(f"    {rank:>2}. {city:<20} : {count:>5} restaurants ({pct:.2f}%)")

    # 5. Top 10 Cuisines
    # (A) Raw combinations
    top_raw_cuisines = df["Cuisines"].value_counts().head(10)
    print("\n[5] Top 10 Cuisine Combinations:")
    for rank, (c_name, count) in enumerate(top_raw_cuisines.items(), 1):
        print(f"    {rank:>2}. {c_name:<32} : {count:>5} restaurants")

    # (B) Individual exploded cuisines
    all_individual = df["Cuisines"].str.split(", ").explode()
    top_individual_cuisines = all_individual.value_counts().head(10)
    print("\n[6] Top 10 Individual Cuisines (Exploded):")
    for rank, (c_name, count) in enumerate(top_individual_cuisines.items(), 1):
        pct = (count / len(df)) * 100
        print(f"    {rank:>2}. {c_name:<20} : {count:>5} occurrences ({pct:.2f}%)")

    # Save outputs
    if save_outputs:
        charts_dir, reports_dir = get_output_paths()

        # Text Report
        report_file = os.path.join(reports_dir, "level_1_task_2_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("COGNIFYZ DATA SCIENCE INTERNSHIP - LEVEL 1 TASK 2 REPORT\n")
            f.write("DESCRIPTIVE ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write("1. NUMERICAL MEASURES:\n")
            f.write(stats_df.to_string(index=False) + "\n\n")
            f.write("2. TOP 10 CITIES BY RESTAURANT COUNT:\n")
            for city, count in top_10_cities.items():
                f.write(f"   - {city:<20}: {count} ({count/len(df)*100:.2f}%)\n")
            f.write("\n3. TOP 10 INDIVIDUAL CUISINES:\n")
            for c_name, count in top_individual_cuisines.items():
                f.write(f"   - {c_name:<20}: {count}\n")
            f.write("\n4. COUNTRY DISTRIBUTION:\n")
            for country, count in country_counts.items():
                f.write(f"   - {country:<22}: {count} ({count/len(df)*100:.2f}%)\n")
        print(f"\n[OUTPUT] Report saved to: {report_file}")

        # Visualizations
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        
        # Chart 1: Top 10 Cities and Top 10 Cuisines
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        sns.barplot(x=top_10_cities.values, y=top_10_cities.index, ax=axes[0], hue=top_10_cities.index, palette="Blues_r", legend=False)
        axes[0].set_title("Top 10 Cities with Highest Restaurant Count", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Number of Restaurants", fontsize=11)
        axes[0].set_ylabel("City", fontsize=11)
        for i, v in enumerate(top_10_cities.values):
            axes[0].text(v + 50, i, f"{v:,}", va="center", fontsize=9, fontweight="bold")

        sns.barplot(x=top_individual_cuisines.values, y=top_individual_cuisines.index, ax=axes[1], hue=top_individual_cuisines.index, palette="Oranges_r", legend=False)
        axes[1].set_title("Top 10 Most Common Cuisines (Individual)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Number of Occurrences", fontsize=11)
        axes[1].set_ylabel("Cuisine", fontsize=11)
        for i, v in enumerate(top_individual_cuisines.values):
            axes[1].text(v + 30, i, f"{v:,}", va="center", fontsize=9, fontweight="bold")

        plt.suptitle("Cognifyz Technologies - Level 1 Task 2: Top Cities and Top Cuisines", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()
        chart_file_1 = os.path.join(charts_dir, "level_1_task_2_cities_and_cuisines.png")
        fig.savefig(chart_file_1, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OUTPUT] Chart saved to: {chart_file_1}")

        # Chart 2: Numerical Distributions
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        
        sns.histplot(df["Price range"], bins=4, ax=axes2[0, 0], color="#457b9d", discrete=True)
        axes2[0, 0].set_title("Price Range Distribution (1 to 4)", fontsize=12, fontweight="bold")
        axes2[0, 0].set_xlabel("Price Range")

        sns.histplot(df[df["Votes"] < 1000]["Votes"], bins=30, ax=axes2[0, 1], color="#2a9d8f", kde=True)
        axes2[0, 1].set_title("Votes Distribution (< 1,000 Votes)", fontsize=12, fontweight="bold")
        axes2[0, 1].set_xlabel("Votes")

        sns.boxplot(x=df["Aggregate rating"], ax=axes2[1, 0], color="#e76f51")
        axes2[1, 0].set_title("Boxplot of Aggregate Rating", fontsize=12, fontweight="bold")
        axes2[1, 0].set_xlabel("Aggregate Rating")

        sns.barplot(x=country_counts.head(7).values, y=country_counts.head(7).index, ax=axes2[1, 1], hue=country_counts.head(7).index, palette="crest", legend=False)
        axes2[1, 1].set_title("Top 7 Countries Distribution", fontsize=12, fontweight="bold")
        axes2[1, 1].set_xlabel("Restaurant Count")

        plt.suptitle("Cognifyz Technologies - Level 1 Task 2: Numerical & Country Distributions", fontsize=14, fontweight="bold")
        plt.tight_layout()
        chart_file_2 = os.path.join(charts_dir, "level_1_task_2_distributions.png")
        fig2.savefig(chart_file_2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"[OUTPUT] Chart saved to: {chart_file_2}")

    print("\n" + "=" * 70)
    print("LEVEL 1 - TASK 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return stats_df


if __name__ == "__main__":
    run_task_2()
