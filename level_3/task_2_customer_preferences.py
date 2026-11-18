"""
Level 3 - Task 2: Customer Preference Analysis
==============================================
Internship: Cognifyz Technologies Data Science Internship
Project: Restaurant Data Analytics & Rating Prediction System

Tasks covered:
1. Analyze the relationship between cuisine types and restaurant ratings.
2. Identify the most popular cuisines based on total customer votes.
3. Identify cuisines that tend to receive the highest average ratings.
4. Present top 10 results with statistical rigor (handling small sample bias).
5. Save analytical report and visualizations.
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
    charts_dir = os.path.join(base_dir, "outputs", "charts")
    reports_dir = os.path.join(base_dir, "outputs", "reports")
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return charts_dir, reports_dir


def run_task_2(save_outputs=True):
    print("=" * 70)
    print("LEVEL 3 - TASK 2: CUSTOMER PREFERENCE ANALYSIS")
    print("=" * 70)

    # 1. Load Data
    data_path = find_data_file("Dataset .csv")
    df = pd.read_csv(data_path, encoding="utf-8")
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")

    # 2. Explode Cuisines
    cuisine_records = []
    for _, row in df.iterrows():
        c_list = [c.strip() for c in str(row["Cuisines"]).split(",")]
        for c in c_list:
            if c and c != "Unknown":
                cuisine_records.append({
                    "Cuisine": c,
                    "Aggregate rating": row["Aggregate rating"],
                    "Votes": row["Votes"],
                    "Price range": row["Price range"]
                })
    c_df = pd.DataFrame(cuisine_records)
    print(f"\n[1] Cuisine Record Expansion:")
    print(f"    - Unique Cuisines Identified: {c_df['Cuisine'].nunique()}")
    print(f"    - Total Cuisine-Restaurant Pairings: {len(c_df):,}")

    # 3. Aggregate Metrics by Cuisine
    cuisine_stats = c_df.groupby("Cuisine").agg(
        Restaurant_Count=("Aggregate rating", "count"),
        Total_Votes=("Votes", "sum"),
        Mean_Votes=("Votes", "mean"),
        Mean_Rating=("Aggregate rating", "mean"),
        Median_Rating=("Aggregate rating", "median")
    )

    # 4. Top 10 Most Popular Cuisines by Votes
    top_10_popular_votes = cuisine_stats.sort_values(by="Total_Votes", ascending=False).head(10)
    print(f"\n[2] Top 10 Most Popular Cuisines (Ranked by Total Votes):")
    print(top_10_popular_votes[["Restaurant_Count", "Total_Votes", "Mean_Rating"]].to_string())

    # 5. Top 10 Highest Rated Cuisines (Min 20 Restaurants for Statistical Reliability)
    # Thresholding avoids 1-restaurant cuisines scoring 4.9 skewing results
    reliable_cuisines = cuisine_stats[cuisine_stats["Restaurant_Count"] >= 20]
    top_10_highest_rated = reliable_cuisines.sort_values(by="Mean_Rating", ascending=False).head(10)
    print(f"\n[3] Top 10 Highest Rated Cuisines (Minimum 20 Restaurants Threshold):")
    print(top_10_highest_rated[["Restaurant_Count", "Total_Votes", "Mean_Rating"]].to_string())

    # 6. Vote-Weighted Rating Analysis
    # Bayesian / Weighted Rating = (v/(v+m)) * R + (m/(v+m)) * C
    C = df["Aggregate rating"].mean()
    m = 100  # vote weight parameter
    cuisine_stats["Weighted_Rating"] = (
        (cuisine_stats["Total_Votes"] / (cuisine_stats["Total_Votes"] + m)) * cuisine_stats["Mean_Rating"] +
        (m / (cuisine_stats["Total_Votes"] + m)) * C
    )
    top_10_weighted = cuisine_stats[cuisine_stats["Restaurant_Count"] >= 20].sort_values(by="Weighted_Rating", ascending=False).head(10)

    # 7. Save Outputs
    if save_outputs:
        charts_dir, reports_dir = get_paths()

        # Save Report
        report_file = os.path.join(reports_dir, "level_3_task_2_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("COGNIFYZ DATA SCIENCE INTERNSHIP - LEVEL 3 TASK 2 REPORT\n")
            f.write("CUSTOMER PREFERENCES & CUISINE DYNAMICS\n")
            f.write("=" * 60 + "\n\n")
            f.write("1. TOP 10 CUISINES BY TOTAL VOTES (POPULARITY):\n")
            f.write(top_10_popular_votes.to_string() + "\n\n")
            f.write("2. TOP 10 CUISINES BY AVERAGE RATING (MIN 20 RESTAURANTS):\n")
            f.write(top_10_highest_rated.to_string() + "\n\n")
            f.write("3. KEY BUSINESS INSIGHTS:\n")
            f.write("   - Volume vs Satisfaction Gap: North Indian and Chinese command massive customer volume\n")
            f.write("     (595k and 364k votes respectively), but their average ratings (2.51 and 2.62) are weighed down\n")
            f.write("     by ubiquitous unrated/average neighborhood eateries in Delhi NCR.\n")
            f.write("   - Premium Niche Cuisines: International, Southern, Vegetarian, Sandwich, Steak, and Sushi boast\n")
            f.write("     the highest customer satisfaction (>3.95 average rating) and higher engagement.\n")
        print(f"\n[OUTPUT] Report saved to: {report_file}")

        # Visualizations
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Chart 1: Top 10 by Votes
        sns.barplot(
            x=top_10_popular_votes["Total_Votes"],
            y=top_10_popular_votes.index,
            ax=axes[0],
            palette="Blues_r"
        )
        axes[0].set_title("Top 10 Most Popular Cuisines by Total Customer Votes", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Total Votes", fontsize=11)
        axes[0].set_ylabel("Cuisine", fontsize=11)
        for i, v in enumerate(top_10_popular_votes["Total_Votes"]):
            axes[0].text(v + 10000, i, f"{v:,}", va="center", fontweight="bold", fontsize=9)

        # Chart 2: Top 10 by Rating
        sns.barplot(
            x=top_10_highest_rated["Mean_Rating"],
            y=top_10_highest_rated.index,
            ax=axes[1],
            palette="Greens_r"
        )
        axes[1].set_title("Top 10 Highest Rated Cuisines (Min 20 Restaurants)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Average Aggregate Rating", fontsize=11)
        axes[1].set_ylabel("Cuisine", fontsize=11)
        axes[1].set_xlim(3.0, 4.5)
        for i, v in enumerate(top_10_highest_rated["Mean_Rating"]):
            axes[1].text(v + 0.03, i, f"{v:.2f} ★", va="center", fontweight="bold", fontsize=9)

        plt.suptitle("Cognifyz Technologies - Level 3 Task 2: Customer Preferences & Cuisine Dynamics", fontsize=15, fontweight="bold")
        plt.tight_layout()
        chart_file = os.path.join(charts_dir, "level_3_task_2_cuisine_preferences.png")
        fig.savefig(chart_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OUTPUT] Chart saved to: {chart_file}")

    print("\n" + "=" * 70)
    print("LEVEL 3 - TASK 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return top_10_popular_votes, top_10_highest_rated


if __name__ == "__main__":
    run_task_2()
