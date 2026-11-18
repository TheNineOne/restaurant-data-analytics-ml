# 📊 Level 1: Data Exploration and Analysis Report
### Cognifyz Technologies Data Science Internship Project
**Project Name:** Restaurant Data Analytics & Rating Prediction System  
**Prepared by:** Senior Data Scientist & QA Lead  
**Dataset Reference:** `Dataset .csv` (9,551 records, 21 columns)

---

## 📌 Executive Summary
Level 1 establishes the foundational data architecture for the restaurant intelligence system. Through systematic preprocessing, descriptive statistical computing, and geospatial spatial analysis, we audited data integrity, diagnosed target variable skewness, examined global market footprints, and validated GPS coordinates across 15 countries.

---

## 🔍 Task 1: Data Exploration & Preprocessing

### 1.1 Dataset Dimensions & Schema Audit
- **Total Record Count:** `9,551` observations.
- **Total Feature Count:** `21` attributes.
- **Identifier Uniqueness:** `Restaurant ID` is $100\%$ unique across all 9,551 rows. Zero redundant entity keys detected.
- **Exact Row Duplication:** `0` duplicate rows detected across the entire attribute matrix.

### 1.2 Missing Value Analysis & Treatment
The dataset exhibits exceptional completeness across 20 of its 21 columns:
- Only `Cuisines` had missing entries: **9 missing rows (0.094%)**.
- **Treatment Strategy:** Given that 9 rows represent less than $0.1\%$ of the dataset, deleting them would unnecessarily remove valid location and financial data. Instead, domain-safe categorical imputation was executed by mapping nulls to `'Unknown'`.

### 1.3 Data Type Standardization
| Column Name | Initial Type | Standardized Type | Purpose |
| :--- | :--- | :--- | :--- |
| `Restaurant ID` | `int64` | `str` | Nominal categorical identifier; prevents arithmetic misuse. |
| `Price range` | `int64` | `int64` | Discrete ordinal ranking (1 to 4). |
| `Aggregate rating` | `float64` | `float64` | Continuous target variable (0.0 to 5.0). |
| `Votes` | `int64` | `int64` | Discrete engagement volume. |

### 1.4 Target Variable (`Aggregate rating`) & Class Imbalance Diagnosis
- **Minimum Rating:** $0.0 ★$
- **Maximum Rating:** $4.9 ★$
- **Overall Mean Rating:** $2.666 ★$
- **Overall Median Rating:** $3.200 ★$
- **Overall Standard Deviation:** $1.516 ★$

#### ⚠️ Critical Finding: Severe Dual-Mode Class Imbalance
- Exactly **2,148 restaurants (22.49%)** carry an `Aggregate rating` of **`0.0`**.
- Cross-tabulation proves these records map exclusively to `Rating text == "Not rated"` and `Rating color == "White"`.
- For actively rated establishments ($>0.0$, 7,403 venues):
  - **Active Mean Rating:** **`3.440 ★`**
  - **Active Median Rating:** **`3.400 ★`**
  - **Distribution:** Approximately normal, centered in the $3.2 - 3.4 ★$ interval.

---

## 📈 Task 2: Descriptive Analysis

### 2.1 Numerical Feature Five-Number Summary
| Feature | Mean | Std Dev | Min | 25% (Q1) | 50% (Median) | 75% (Q3) | Max | IQR | Skewness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Average Cost for two** | 1,199.21 | 16,121.18 | 0.00 | 250.00 | 400.00 | 700.00 | 800,000.00 | 450.00 | +35.48 |
| **Price range** | 1.80 | 0.91 | 1.00 | 1.00 | 2.00 | 2.00 | 4.00 | 1.00 | +0.89 |
| **Aggregate rating** | 2.67 | 1.52 | 0.00 | 2.50 | 3.20 | 3.70 | 4.90 | 1.20 | -0.95 |
| **Votes** | 156.91 | 430.17 | 0.00 | 5.00 | 31.00 | 131.00 | 10,934.00 | 126.00 | +8.81 |

*Domain Note on Cost Outliers:* The extreme maximum cost of $800,000$ originates from restaurants in Jakarta, Indonesia where prices are quoted in Indonesian Rupiah (IDR). $800,000$ IDR equates to approximately $\$50$ USD, which is a standard dining cost.

### 2.2 Top 10 Cities by Restaurant Count
1. **New Delhi:** $5,473$ ($57.30\%$)
2. **Gurgaon:** $1,118$ ($11.71\%$)
3. **Noida:** $1,080$ ($11.31\%$)
4. **Faridabad:** $251$ ($2.63\%$)
5. **Ghaziabad:** $25$ ($0.26\%$)
6. **Ahmedabad:** $21$ ($0.22\%$)
7. **Amritsar:** $21$ ($0.22\%$)
8. **Bhubaneshwar:** $21$ ($0.22\%$)
9. **Guwahati:** $21$ ($0.22\%$)
10. **Lucknow:** $21$ ($0.22\%$)

*Takeaway:* Over $83\%$ of the entire global dataset is concentrated within the Delhi National Capital Region (NCR).

### 2.3 Top 10 Individual Cuisines (Exploded Representation)
1. **North Indian:** $3,960$ occurrences ($41.46\%$)
2. **Chinese:** $2,735$ occurrences ($28.64\%$)
3. **Fast Food:** $1,986$ occurrences ($20.79\%$)
4. **Mughlai:** $995$ occurrences ($10.42\%$)
5. **Italian:** $764$ occurrences ($8.00\%$)
6. **Bakery:** $745$ occurrences ($7.80\%$)
7. **Continental:** $736$ occurrences ($7.71\%$)
8. **Cafe:** $703$ occurrences ($7.36\%$)
9. **Desserts:** $653$ occurrences ($6.84\%$)
10. **South Indian:** $636$ occurrences ($6.66\%$)

---

## 🗺️ Task 3: Geospatial Analysis

### 3.1 Geographic Coordinate Audit & Anomaly Detection
- **Valid GPS Records:** $9,054$ restaurants ($94.80\%$).
- **Missing / Zero GPS Coordinates `(0.0, 0.0)`:** $497$ restaurants ($5.20\%$).
- **Resolution:** All 497 invalid coordinates belong to regional addresses in Delhi NCR. During geographical rendering, these zero coordinates are filtered out to prevent clustering at Null Island ($0^\circ\text{N}, 0^\circ\text{E}$ off the coast of Africa).

### 3.2 Correlation Between Location and Rating
- **Pearson Correlation (Latitude vs Aggregate rating):** $r = -0.1246$ ($p = 1.19 \times 10^{-32}$)
- **Pearson Correlation (Longitude vs Aggregate rating):** $r = -0.2062$ ($p = 1.74 \times 10^{-87}$)
- **Spearman Rank Correlation:** $\rho_{lat} = -0.1126$, $\rho_{lon} = -0.1922$
- **Inference:** While linear spatial correlation is weakly negative across global degrees, localized market differences are prominent:
  - Western markets (United States, United Kingdom, UAE, New Zealand) average $>4.0 ★$ ratings with virtually zero unrated venues.
  - The high concentration of unrated budget vendors in Delhi NCR pulls down the regional average to $2.44 ★$.

### 3.3 Interactive Folium Map Delivery
- An interactive, cluster-enabled map was generated and saved to `outputs/charts/restaurant_geospatial_map.html`.
- Implements `MarkerCluster` with rating-calibrated color pins (Dark Green, Green, Orange, Red, Gray) and informative HTML popups detailing restaurant name, cuisine offerings, average cost, rating tier, and vote volume.

---

## 🏁 Level 1 Conclusion
All exploratory and preprocessing milestones are complete. The dataset is validated, clean, and ready for advanced business analysis in Level 2.
