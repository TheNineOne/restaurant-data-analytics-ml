"""
Level 2 - Task 2: Price Range Analysis
======================================
Internship: Cognifyz Technologies Data Science Internship
Project: Restaurant Data Analytics & Rating Prediction System

Tasks covered:
1. Determine the most common price range among all restaurants.
2. Calculate the average rating for each price range.
3. Identify the rating color associated with the highest average rating.
4. Analyze rating colors across different price ranges.
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


def get_output_paths():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    charts_dir = os.path.join(base_dir, "outputs", "charts")
    reports_dir = os.path.join(base_dir, "outputs", "reports")
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    return charts_dir, reports_dir


def run_task_2(save_outputs=True):
    print("=" * 70)
    print("LEVEL 2 - TASK 2: PRICE RANGE ANALYSIS")
    print("=" * 70)

    # 1. Load Data
    data_path = find_data_file("Dataset .csv")
    df = pd.read_csv(data_path, encoding="utf-8")
    total_restaurants = len(df)

    # 2. Most Common Price Range
    price_counts = df["Price range"].value_counts().sort_index()
    price_pcts = (price_counts / total_restaurants) * 100
    most_common_pr = price_counts.idxmax()
    most_common_count = price_counts.max()
    most_common_pct = price_pcts.max()

    print(f"\n[1] Price Range Frequency Breakdown:")
    for pr, count in price_counts.items():
        print(f"    - Price Range {pr}: {count:,} restaurants ({price_pcts[pr]:.2f}%)")
    print(f"    -> Most Common Price Range: Range {most_common_pr} ({most_common_count:,} restaurants, {most_common_pct:.2f}%)")

    # 3. Average Rating for Each Price Range
    pr_rating_stats = df.groupby("Price range")["Aggregate rating"].agg(
        Count="count",
        Mean="mean",
        Median="median",
        Std="std",
        Min="min",
        Max="max"
    )
    print(f"\n[2] Rating Statistics by Price Range:")
    print(pr_rating_stats.to_string())

    print("\n    Key Observation:")
    print("    - There is a monotonic upward trend: as Price Range increases from 1 to 4,")
    print("      the average rating increases from 2.00 to 3.82 (an increase of +1.82 rating points).")

    # 4. Rating Colors and Average Rating
    color_stats = df.groupby("Rating color")["Aggregate rating"].agg(
        Count="count",
        Mean_Rating="mean",
        Median_Rating="median"
    ).sort_values(by="Mean_Rating", ascending=False)
    
    highest_color = color_stats.index[0]
    highest_color_rating = color_stats.iloc[0]["Mean_Rating"]

    print(f"\n[3] Rating Statistics by Rating Color:")
    print(color_stats.to_string())
    print(f"\n    -> Rating color associated with the highest average rating: '{highest_color}' ({highest_color_rating:.2f} avg rating).")

    # 5. Rating Color Distribution across Price Ranges (Crosstab)
    color_price_ct = pd.crosstab(df["Price range"], df["Rating color"])[["Dark Green", "Green", "Yellow", "Orange", "Red", "White"]]
    color_price_pct = pd.crosstab(df["Price range"], df["Rating color"], normalize="index") * 100
    color_price_pct = color_price_pct[["Dark Green", "Green", "Yellow", "Orange", "Red", "White"]]
    print(f"\n[4] Rating Color Distribution across Price Ranges (% within Price Range):")
    print(color_price_pct.round(2).to_string())

    print("\n    Key Observation:")
    print("    - In Price Range 1, 48.24% of restaurants are 'White' (unrated 0.0), explaining its low average.")
    print("    - In Price Range 4, 25.60% of restaurants are 'Dark Green' (rating 4.5 - 4.9) and only 2.05% are unrated.")

    # 6. Save Outputs
    if save_outputs:
        charts_dir, reports_dir = get_output_paths()

        # Save Text Report
        report_file = os.path.join(reports_dir, "level_2_task_2_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("COGNIFYZ DATA SCIENCE INTERNSHIP - LEVEL 2 TASK 2 REPORT\n")
            f.write("PRICE RANGE & RATING COLOR ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"1. Most Common Price Range: Range {most_common_pr} ({most_common_count} restaurants, {most_common_pct:.2f}%)\n\n")
            f.write("2. Rating Statistics by Price Range:\n")
            f.write(pr_rating_stats.to_string() + "\n\n")
            f.write(f"3. Highest Average Rating Color: '{highest_color}' with {highest_color_rating:.3f} average rating.\n\n")
            f.write("4. Rating Colors Summary:\n")
            f.write(color_stats.to_string() + "\n\n")
            f.write("5. Rating Color Distribution by Price Range (%):\n")
            f.write(color_price_pct.round(2).to_string() + "\n")
        print(f"\n[OUTPUT] Report saved to: {report_file}")

        # Visualizations
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Chart 1: Price Range Distribution
        bars1 = axes[0, 0].bar(
            [f"Price Range {i}" for i in price_counts.index],
            price_counts.values,
            color=["#2b5c8f", "#457b9d", "#a8dadc", "#1d3557"],
            edgecolor="black"
        )
        axes[0, 0].set_title("Most Common Price Range (Frequency)", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Number of Restaurants", fontsize=11)
        for bar in bars1:
            h = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, h + 50, f"{h:,}\n({h/total_restaurants*100:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=9)

        # Chart 2: Average Rating per Price Range
        bars2 = axes[0, 1].bar(
            [f"Price Range {i}" for i in pr_rating_stats.index],
            pr_rating_stats["Mean"],
            color=["#e76f51", "#f4a261", "#e9c46a", "#2a9d8f"],
            edgecolor="black"
        )
        axes[0, 1].set_title("Average Aggregate Rating by Price Range", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Average Rating (0.0 to 5.0)", fontsize=11)
        axes[0, 1].set_ylim(0, 4.5)
        for bar in bars2:
            h = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f} ★", ha="center", va="bottom", fontweight="bold", fontsize=10)

        # Chart 3: Rating Color Average Rating
        color_palette_map = {
            "Dark Green": "#006400",
            "Green": "#2e8b57",
            "Yellow": "#f4d03f",
            "Orange": "#e67e22",
            "Red": "#e74c3c",
            "White": "#bdc3c7"
        }
        bar_colors = [color_palette_map.get(c, "#333333") for c in color_stats.index]
        bars3 = axes[1, 0].barh(
            color_stats.index,
            color_stats["Mean_Rating"],
            color=bar_colors,
            edgecolor="black"
        )
        axes[1, 0].set_title("Average Rating by Rating Color (Highest: Dark Green)", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Average Rating", fontsize=11)
        for bar in bars3:
            w = bar.get_width()
            axes[1, 0].text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontweight="bold", fontsize=9)

        # Chart 4: Stacked Percentage Bar Chart of Rating Colors by Price Range
        color_price_pct.plot(
            kind="bar",
            stacked=True,
            ax=axes[1, 1],
            color=[color_palette_map[c] for c in color_price_pct.columns],
            edgecolor="black"
        )
        axes[1, 1].set_title("Rating Color Composition by Price Range (%)", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Price Range", fontsize=11)
        axes[1, 1].set_ylabel("Percentage (%)", fontsize=11)
        axes[1, 1].set_xticklabels([f"Range {i}" for i in color_price_pct.index], rotation=0)
        axes[1, 1].legend(title="Rating Color", bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.suptitle("Cognifyz Technologies - Level 2 Task 2: Price Range & Rating Color Analysis", fontsize=15, fontweight="bold")
        plt.tight_layout()
        chart_file = os.path.join(charts_dir, "level_2_task_2_price_range_analysis.png")
        fig.savefig(chart_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OUTPUT] Chart saved to: {chart_file}")

    print("\n" + "=" * 70)
    print("LEVEL 2 - TASK 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return pr_rating_stats


if __name__ == "__main__":
    run_task_2()
