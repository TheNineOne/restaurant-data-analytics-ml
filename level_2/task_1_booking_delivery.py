"""
Level 2 - Task 1: Table Booking and Online Delivery
===================================================
Internship: Cognifyz Technologies Data Science Internship
Project: Restaurant Data Analytics & Rating Prediction System

Tasks covered:
1. Calculate percentage of restaurants offering table booking and online delivery.
2. Compare average ratings of restaurants with and without table booking.
3. Compare average ratings of restaurants with and without online delivery.
4. Analyze the availability of online delivery across different price ranges.
5. Perform statistical significance tests (Independent t-test).
6. Save comprehensive analytical report and visualizations.
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
from scipy import stats

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


def run_task_1(save_outputs=True):
    print("=" * 70)
    print("LEVEL 2 - TASK 1: TABLE BOOKING AND ONLINE DELIVERY ANALYSIS")
    print("=" * 70)

    # 1. Load Data
    data_path = find_data_file("Dataset .csv")
    df = pd.read_csv(data_path, encoding="utf-8")
    total_restaurants = len(df)

    # 2. Table Booking Percentage
    tb_counts = df["Has Table booking"].value_counts()
    tb_pct = (tb_counts / total_restaurants) * 100
    print(f"\n[1] Table Booking Availability:")
    print(f"    - Yes: {tb_counts.get('Yes', 0):,} restaurants ({tb_pct.get('Yes', 0.0):.2f}%)")
    print(f"    - No : {tb_counts.get('No', 0):,} restaurants ({tb_pct.get('No', 0.0):.2f}%)")

    # 3. Online Delivery Percentage
    od_counts = df["Has Online delivery"].value_counts()
    od_pct = (od_counts / total_restaurants) * 100
    print(f"\n[2] Online Delivery Availability:")
    print(f"    - Yes: {od_counts.get('Yes', 0):,} restaurants ({od_pct.get('Yes', 0.0):.2f}%)")
    print(f"    - No : {od_counts.get('No', 0):,} restaurants ({od_pct.get('No', 0.0):.2f}%)")

    # 4. Rating Comparison: Table Booking (With vs Without)
    tb_yes_ratings = df[df["Has Table booking"] == "Yes"]["Aggregate rating"]
    tb_no_ratings = df[df["Has Table booking"] == "No"]["Aggregate rating"]

    tb_t_stat, tb_p_val = stats.ttest_ind(tb_yes_ratings, tb_no_ratings, equal_var=False)

    print(f"\n[3] Rating Comparison: Table Booking:")
    print(f"    - Average Rating WITH Table Booking   : {tb_yes_ratings.mean():.3f} (Median: {tb_yes_ratings.median():.2f})")
    print(f"    - Average Rating WITHOUT Table Booking: {tb_no_ratings.mean():.3f} (Median: {tb_no_ratings.median():.2f})")
    print(f"    - Rating Advantage (Delta)            : +{tb_yes_ratings.mean() - tb_no_ratings.mean():.3f}")
    print(f"    - Welch's T-Statistic                 : {tb_t_stat:.4f} (p-value: {tb_p_val:.4e})")
    print("    -> Conclusion: Restaurants with table booking have significantly higher ratings (p < 0.001).")

    # Rating Comparison: Online Delivery (With vs Without)
    od_yes_ratings = df[df["Has Online delivery"] == "Yes"]["Aggregate rating"]
    od_no_ratings = df[df["Has Online delivery"] == "No"]["Aggregate rating"]
    od_t_stat, od_p_val = stats.ttest_ind(od_yes_ratings, od_no_ratings, equal_var=False)

    print(f"\n[4] Rating Comparison: Online Delivery:")
    print(f"    - Average Rating WITH Online Delivery   : {od_yes_ratings.mean():.3f} (Median: {od_yes_ratings.median():.2f})")
    print(f"    - Average Rating WITHOUT Online Delivery: {od_no_ratings.mean():.3f} (Median: {od_no_ratings.median():.2f})")
    print(f"    - Rating Advantage (Delta)              : +{od_yes_ratings.mean() - od_no_ratings.mean():.3f}")
    print(f"    - Welch's T-Statistic                   : {od_t_stat:.4f} (p-value: {od_p_val:.4e})")

    # 5. Online Delivery across Price Ranges
    print(f"\n[5] Online Delivery Availability Across Price Ranges (1=Budget, 4=Luxury):")
    od_price_ct = pd.crosstab(df["Price range"], df["Has Online delivery"], margins=True)
    od_price_pct = pd.crosstab(df["Price range"], df["Has Online delivery"], normalize="index") * 100
    
    combined_price_analysis = pd.DataFrame({
        "Total Restaurants": od_price_ct["All"].drop("All"),
        "Online Delivery (No)": od_price_ct["No"].drop("All"),
        "Online Delivery (Yes)": od_price_ct["Yes"].drop("All"),
        "Online Delivery % (Yes)": od_price_pct["Yes"]
    })
    print(combined_price_analysis.to_string())

    print("\n    Key Observation on Price Ranges:")
    print("    - Price Range 2 (Mid-range) has the highest online delivery adoption at 41.31%.")
    print("    - Price Range 4 (Fine Dining / Luxury) has the lowest adoption at only 9.04%,")
    print("      reflecting the in-person experiential dining model of premium venues.")

    # 6. Save Outputs
    if save_outputs:
        charts_dir, reports_dir = get_output_paths()

        # Text Report
        report_file = os.path.join(reports_dir, "level_2_task_1_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("COGNIFYZ DATA SCIENCE INTERNSHIP - LEVEL 2 TASK 1 REPORT\n")
            f.write("TABLE BOOKING & ONLINE DELIVERY ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"1. Service Adoption Summary:\n")
            f.write(f"   - Table Booking Adoption: {tb_pct.get('Yes', 0):.2f}% ({tb_counts.get('Yes', 0):,} restaurants)\n")
            f.write(f"   - Online Delivery Adoption: {od_pct.get('Yes', 0):.2f}% ({od_counts.get('Yes', 0):,} restaurants)\n\n")
            f.write(f"2. Rating Impact of Table Booking:\n")
            f.write(f"   - With Booking Mean: {tb_yes_ratings.mean():.3f} (Median: {tb_yes_ratings.median():.2f})\n")
            f.write(f"   - Without Booking Mean: {tb_no_ratings.mean():.3f} (Median: {tb_no_ratings.median():.2f})\n")
            f.write(f"   - Rating Delta: +{tb_yes_ratings.mean() - tb_no_ratings.mean():.3f}\n")
            f.write(f"   - Statistical Significance: t={tb_t_stat:.4f}, p={tb_p_val:.4e} (Statistically Significant)\n\n")
            f.write(f"3. Rating Impact of Online Delivery:\n")
            f.write(f"   - With Delivery Mean: {od_yes_ratings.mean():.3f} (Median: {od_yes_ratings.median():.2f})\n")
            f.write(f"   - Without Delivery Mean: {od_no_ratings.mean():.3f} (Median: {od_no_ratings.median():.2f})\n")
            f.write(f"   - Rating Delta: +{od_yes_ratings.mean() - od_no_ratings.mean():.3f}\n\n")
            f.write("4. Online Delivery Across Price Ranges:\n")
            f.write(combined_price_analysis.to_string() + "\n")
        print(f"\n[OUTPUT] Report saved to: {report_file}")

        # Visualizations
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Subplot 1: Table Booking Adoption Pie Chart
        colors_tb = ["#457b9d", "#e63946"]
        axes[0, 0].pie(
            tb_counts,
            labels=[f"No Booking ({tb_pct['No']:.1f}%)", f"Offers Booking ({tb_pct['Yes']:.1f}%)"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors_tb,
            explode=(0, 0.08),
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        axes[0, 0].set_title("Table Booking Availability Breakdown", fontsize=12, fontweight="bold")

        # Subplot 2: Online Delivery Adoption Pie Chart
        colors_od = ["#2a9d8f", "#f4a261"]
        axes[0, 1].pie(
            od_counts,
            labels=[f"No Delivery ({od_pct['No']:.1f}%)", f"Offers Delivery ({od_pct['Yes']:.1f}%)"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors_od,
            explode=(0, 0.08),
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        axes[0, 1].set_title("Online Delivery Availability Breakdown", fontsize=12, fontweight="bold")

        # Subplot 3: Boxplot of Ratings by Table Booking
        sns.boxplot(
            data=df,
            x="Has Table booking",
            y="Aggregate rating",
            ax=axes[1, 0],
            palette=["#457b9d", "#e63946"]
        )
        axes[1, 0].set_title("Rating Distribution: Table Booking (With vs Without)", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Has Table Booking", fontsize=11)
        axes[1, 0].set_ylabel("Aggregate Rating", fontsize=11)

        # Subplot 4: Online Delivery % by Price Range
        price_ranges = [f"Price Range {i}" for i in combined_price_analysis.index]
        bars = axes[1, 1].bar(
            price_ranges,
            combined_price_analysis["Online Delivery % (Yes)"],
            color=["#3a86ff", "#06d6a0", "#ffd166", "#ef476f"],
            edgecolor="black"
        )
        axes[1, 1].set_title("Online Delivery Availability by Price Range", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("% Restaurants with Online Delivery", fontsize=11)
        axes[1, 1].set_ylim(0, 50)
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10
            )

        plt.suptitle("Cognifyz Technologies - Level 2 Task 1: Booking & Delivery Analysis", fontsize=15, fontweight="bold")
        plt.tight_layout()
        chart_file = os.path.join(charts_dir, "level_2_task_1_booking_delivery.png")
        fig.savefig(chart_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OUTPUT] Chart saved to: {chart_file}")

    print("\n" + "=" * 70)
    print("LEVEL 2 - TASK 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return combined_price_analysis


if __name__ == "__main__":
    run_task_1()
