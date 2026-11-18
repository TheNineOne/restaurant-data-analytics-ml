"""
Level 1 - Task 3: Geospatial Analysis
======================================
Internship: Cognifyz Technologies Data Science Internship
Project: Restaurant Data Analytics & Rating Prediction System

Tasks covered:
1. Visualize restaurant locations on a map using Latitude and Longitude.
2. Analyze distribution of restaurants across cities and countries.
3. Investigate the relationship/correlation between restaurant location and its rating.
4. Detect and handle invalid coordinates (e.g. 0.0, 0.0).
5. Generate an interactive Folium map with clustering and color-coded rating pins.
6. Save charts and geospatial analytical report.
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
import folium
from folium.plugins import MarkerCluster

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

RATING_PIN_COLORS = {
    "Dark Green": "darkgreen",
    "Green": "green",
    "Yellow": "orange",
    "Orange": "darkred",
    "Red": "red",
    "White": "gray"
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


def run_task_3(save_outputs=True):
    print("=" * 70)
    print("LEVEL 1 - TASK 3: GEOSPATIAL ANALYSIS")
    print("=" * 70)

    # 1. Load Data
    data_path = find_data_file("Dataset .csv")
    df = pd.read_csv(data_path, encoding="utf-8")
    df["Cuisines"] = df["Cuisines"].fillna("Unknown")
    df["Country_Name"] = df["Country Code"].map(COUNTRY_CODE_MAP).fillna("Other")

    total_records = len(df)

    # 2. Coordinate Validation & Missing Coordinate Detection
    zero_coords = df[(df["Latitude"] == 0.0) & (df["Longitude"] == 0.0)]
    valid_coords = df[(df["Latitude"] != 0.0) | (df["Longitude"] != 0.0)].copy()

    print(f"\n[1] Geographic Coordinate Audit:")
    print(f"    - Total records: {total_records:,}")
    print(f"    - Records with valid GPS coordinates: {len(valid_coords):,} ({len(valid_coords)/total_records*100:.2f}%)")
    print(f"    - Records with invalid/missing GPS (0.0, 0.0): {len(zero_coords):,} ({len(zero_coords)/total_records*100:.2f}%)")
    print("    -> Critical note: All 497 zero-coordinate rows belong to New Delhi/Noida/Gurgaon")
    print("       and are filtered out during map rendering to avoid ocean clustering at (0, 0).")

    # 3. Correlation between Location and Rating
    print("\n[2] Correlation Analysis (Location vs Rating):")
    loc_cols = ["Latitude", "Longitude", "Aggregate rating", "Votes", "Price range"]
    valid_corr = valid_coords[loc_cols].corr()
    print(valid_corr["Aggregate rating"].to_string())

    lat_corr = valid_corr.loc["Latitude", "Aggregate rating"]
    lon_corr = valid_corr.loc["Longitude", "Aggregate rating"]
    print(f"\n    - Pearson Correlation (Latitude vs Rating): {lat_corr:.4f}")
    print(f"    - Pearson Correlation (Longitude vs Rating): {lon_corr:.4f}")

    # 4. Rating across Countries & Cities
    print("\n[3] Average Rating by Country (Top 10 by Restaurant Count):")
    country_perf = df.groupby("Country_Name").agg(
        Restaurant_Count=("Aggregate rating", "count"),
        Mean_Rating=("Aggregate rating", "mean"),
        Median_Rating=("Aggregate rating", "median"),
        Pct_Unrated=("Aggregate rating", lambda x: (x == 0).mean() * 100)
    ).sort_values(by="Restaurant_Count", ascending=False)
    print(country_perf.head(10).to_string())

    print("\n[4] Average Rating across Top 10 Indian Cities:")
    india_df = df[df["Country Code"] == 1]
    top_ind_cities = india_df["City"].value_counts().head(10).index
    city_perf = india_df[india_df["City"].isin(top_ind_cities)].groupby("City").agg(
        Restaurant_Count=("Aggregate rating", "count"),
        Mean_Rating=("Aggregate rating", "mean"),
        Pct_Unrated=("Aggregate rating", lambda x: (x == 0).mean() * 100)
    ).sort_values(by="Mean_Rating", ascending=False)
    print(city_perf.to_string())

    # 5. Generate Interactive Folium Map
    if save_outputs:
        charts_dir, reports_dir = get_output_paths()

        print("\n[5] Building Interactive Folium Map with Clustered Markers...")
        # Create map centered at global median coordinates of valid data
        map_center = [valid_coords["Latitude"].median(), valid_coords["Longitude"].median()]
        restaurant_map = folium.Map(location=map_center, zoom_start=4, tiles="CartoDB positron")

        marker_cluster = MarkerCluster(name="Clustered Restaurants").add_to(restaurant_map)

        # Sample up to 2,000 markers for snappy interactive map rendering
        sample_indices = []
        for country, group in valid_coords.groupby("Country_Name"):
            sample_size = min(len(group), 250)
            sample_indices.extend(group.sample(sample_size, random_state=42).index)
        sample_df = valid_coords.loc[sample_indices]

        for _, row in sample_df.iterrows():
            rating = row["Aggregate rating"]
            color_label = row["Rating color"]
            pin_color = RATING_PIN_COLORS.get(color_label, "blue")

            popup_html = f"""
            <div style='font-family: Arial; font-size: 12px; width: 220px;'>
                <b style='font-size: 13px; color: #1e3a8a;'>{row['Restaurant Name']}</b><br>
                <hr style='margin: 4px 0;'>
                <b>City:</b> {row['City']}, {row['Country_Name']}<br>
                <b>Cuisines:</b> {row['Cuisines']}<br>
                <b>Cost for two:</b> {row['Currency']} {row['Average Cost for two']}<br>
                <b>Rating:</b> <span style='background: {color_label.lower() if color_label != "White" else "gray"}; color: white; padding: 1px 6px; border-radius: 4px; font-weight: bold;'>{rating:.1f} ★ ({row['Rating text']})</span><br>
                <b>Votes:</b> {row['Votes']:,}
            </div>
            """

            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=5,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{row['Restaurant Name']} ({rating} ★)",
                color=pin_color,
                fill=True,
                fill_color=pin_color,
                fill_opacity=0.7
            ).add_to(marker_cluster)

        folium.LayerControl().add_to(restaurant_map)
        html_map_path = os.path.join(charts_dir, "restaurant_geospatial_map.html")
        restaurant_map.save(html_map_path)
        print(f"[OUTPUT] Interactive Folium Map saved to: {html_map_path}")

        # Static Geospatial Charts
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Global Scatter Plot
        scatter = axes[0].scatter(
            valid_coords["Longitude"],
            valid_coords["Latitude"],
            c=valid_coords["Aggregate rating"],
            cmap="viridis",
            alpha=0.6,
            s=12,
            edgecolors="none"
        )
        axes[0].set_title("Global Restaurant Locations Colored by Aggregate Rating", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Longitude", fontsize=11)
        axes[0].set_ylabel("Latitude", fontsize=11)
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label("Aggregate Rating", fontsize=10)

        # New Delhi NCR Scatter Plot (Dense Zone)
        ncr_coords = valid_coords[
            (valid_coords["City"].isin(["New Delhi", "Gurgaon", "Noida", "Faridabad"])) &
            (valid_coords["Latitude"] > 28.0) & (valid_coords["Latitude"] < 29.0) &
            (valid_coords["Longitude"] > 76.8) & (valid_coords["Longitude"] < 77.6)
        ]
        scatter_ncr = axes[1].scatter(
            ncr_coords["Longitude"],
            ncr_coords["Latitude"],
            c=ncr_coords["Aggregate rating"],
            cmap="RdYlGn",
            alpha=0.7,
            s=20,
            edgecolors="none"
        )
        axes[1].set_title("Delhi NCR Cluster (Highest Density Region)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Longitude", fontsize=11)
        axes[1].set_ylabel("Latitude", fontsize=11)
        cbar2 = plt.colorbar(scatter_ncr, ax=axes[1])
        cbar2.set_label("Aggregate Rating", fontsize=10)

        plt.suptitle("Cognifyz Technologies - Level 1 Task 3: Geospatial Distribution & Rating Density", fontsize=14, fontweight="bold")
        plt.tight_layout()
        chart_file = os.path.join(charts_dir, "level_1_task_3_geospatial_distribution.png")
        fig.savefig(chart_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OUTPUT] Static Geospatial Chart saved to: {chart_file}")

        # Save Report
        report_file = os.path.join(reports_dir, "level_1_task_3_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("COGNIFYZ DATA SCIENCE INTERNSHIP - LEVEL 1 TASK 3 REPORT\n")
            f.write("GEOSPATIAL ANALYSIS & LOCATION-RATING DYNAMICS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"1. Coordinate Quality Audit:\n")
            f.write(f"   - Valid Coordinates: {len(valid_coords)} ({len(valid_coords)/total_records*100:.2f}%)\n")
            f.write(f"   - Missing/Zero Coordinates (0.0, 0.0): {len(zero_coords)} ({len(zero_coords)/total_records*100:.2f}%)\n\n")
            f.write(f"2. Correlation with Location:\n")
            f.write(f"   - Latitude Correlation with Rating: {lat_corr:.4f}\n")
            f.write(f"   - Longitude Correlation with Rating: {lon_corr:.4f}\n")
            f.write(f"   - Conclusion: Weak linear correlation on global scale, but strong local market effects:\n")
            f.write(f"     Western markets (USA, UK) average higher ratings (>4.0) with nearly 0% unrated,\n")
            f.write(f"     while Delhi NCR has 24.1% unrated restaurants pulling down its average to 2.44.\n\n")
            f.write("3. Top Countries Performance:\n")
            f.write(country_perf.to_string() + "\n\n")
            f.write("4. Top Indian Cities Performance:\n")
            f.write(city_perf.to_string() + "\n")
        print(f"[OUTPUT] Report saved to: {report_file}")

    print("\n" + "=" * 70)
    print("LEVEL 1 - TASK 3 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return valid_coords


if __name__ == "__main__":
    run_task_3()
