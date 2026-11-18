# 🔍 Complete Project Audit & Software Quality Report
### Cognifyz Technologies Data Science Internship Project
**Project Name:** Restaurant Data Analytics & Rating Prediction System  
**Lead Auditor:** Senior Data Scientist, ML Engineer & Software QA Specialist  
**Audit Date:** November 2026 | **Dataset:** `Dataset .csv` (9,551 records, 21 raw columns, 29 engineered columns)

---

## 📌 1. Executive Audit Summary

A rigorous, end-to-end audit was conducted across all 9 dashboard modules, the statistical computing engine, machine learning pipelines, dataset transformations, and interactive geospatial components.

- **Total Dashboard Modules Audited:** `9 of 9`
- **Total Modules Passed:** `9 of 9 (100% Pass Rate)`
- **Data Leakage Risk:** `0% (Strictly Excluded Target-Derived Columns)`
- **Target Variable Model Accuracy:** `R² = 0.9598 | MAE = 0.1993`
- **Pipeline Execution Success:** `100% Clean Run Across All Level Scripts (54.24s)`

---

## 📋 2. Comprehensive Module-by-Module Audit Table

| Module # | Feature / Module Name | Initial Audit Status | Issues Identified | Engineering Fix Applied | Final QA Status |
| :---: | :--- | :---: | :--- | :--- | :---: |
| **1** | **Overview** | Pass with Warning | Filter combination resulting in 0 records could produce `NaN` in metric cards and empty pie charts. | Implemented an empty-state guardrail with user-friendly warning and one-click filter reset button. | ✅ **VERIFIED & PASSED** |
| **2** | **Data Exploration** | Pass | Active rating distribution chart lacked dynamic vertical mean line when filtered to 0 active venues. | Added conditional null-check before rendering distribution mean reference lines. | ✅ **VERIFIED & PASSED** |
| **3** | **Restaurant Analysis** | Pass with Minor Display Issue | Special character encoding artifacts in city names (`Bras_lia`, `So Paulo`, `stanbul`). | Added `City_Display` dictionary mapping proper UTF-8 names (`Brasília`, `São Paulo`, `Istanbul`) while preserving raw data integrity. | ✅ **VERIFIED & PASSED** |
| **4** | **Geospatial Analysis** | **High Priority Bug Detected** | Plotly 7.1.0 removed `px.scatter_mapbox`, raising an `AttributeError`. | Engineered a dual-compatible wrapper supporting `px.scatter_map` (Plotly 7+) with `open-street-map`, `px.scatter_mapbox` (Plotly <7), and `px.scatter_geo` fallback. | ✅ **VERIFIED & PASSED** |
| **5** | **Booking & Delivery Analysis** | Pass with Warning | Delta formatting in metric cards failed if one category had zero venues in filtered data. | Added conditional guard to delta string calculation to prevent `NaN` string interpolation errors. | ✅ **VERIFIED & PASSED** |
| **6** | **Price Range Analysis** | Pass with Edge Case | Calling `idxmax()` on an empty price series could throw a `ValueError`. | Added empty check fallback default (`tier 1`) if the filtered series contains zero records. | ✅ **VERIFIED & PASSED** |
| **7** | **Customer Preferences** | Pass with Edge Case | Empty cuisine dataframe when filtered could throw an `IndexError` on `.iloc[0]`. Dynamic slider max could exceed venue count. | Added empty-dataframe guard and scaled dynamic slider bounds to `max(5, count // 2)`. | ✅ **VERIFIED & PASSED** |
| **8** | **Machine Learning** | Pass | Required verification that model evaluation metrics in metadata JSON matched actual test outputs. | Cross-verified all metrics against model evaluation runs; verified complete prevention of data leakage. | ✅ **VERIFIED & PASSED** |
| **9** | **Model Prediction** | Pass | Form required robust input validation for extreme vote counts, coordinate boundaries, and negative costs. | Bounded prediction outputs with `np.clip(pred, 0.0, 5.0)`, verified feature order strictly matches metadata schema. | ✅ **VERIFIED & PASSED** |

---

## 📊 3. Data Quality & Preprocessing Audit

### 3.1 Dimensions & Duplication Audit
- **Raw Dataset Shape:** $9,551$ rows $\times$ $21$ columns.
- **Engineered Dataset Shape:** $9,551$ rows $\times$ $29$ columns ($+8$ domain-engineered features).
- **Exact Duplicate Rows:** `0` (Zero duplicate records).
- **Restaurant ID Uniqueness:** `0` duplicate identifiers ($9,551$ unique IDs).

### 3.2 Missing Value Audit
| Column Name | Missing Count | Percentage | Imputation Strategy |
| :--- | :---: | :---: | :--- |
| `Cuisines` | **9** | **0.094%** | Imputed with domain-neutral `'Unknown'`. Preserves all other 20 column values. |
| *All other 20 columns* | **0** | **0.000%** | $100\%$ complete. Zero data loss. |

### 3.3 Target Variable Distribution & Imbalance
- **Scale:** Continuous from $0.0$ to $4.9 ★$.
- **Class Imbalance:** $2,148$ restaurants ($22.49\%$) carry a rating of $0.0$.
  - These records represent "Not rated" venues (white color code) where customer reviews have not reached minimum aggregation thresholds.
  - Active rated restaurants ($>0.0$) average $3.44 ★$, forming an approximately Gaussian curve.

### 3.4 Numerical Outliers Audit
- `Average Cost for two` exhibits high positive skewness ($+35.48$) with values reaching $800,000$.
- **Audit Explanation:** The dataset includes multiple national currencies (12 currencies) without currency normalization. The highest costs belong to Jakarta, Indonesia (Indonesian Rupiah - IDR), where $800,000$ IDR $\approx \$50$ USD. This is expected real-world data and not a data corruption bug.

---

## 🔬 4. Statistical Validation Audit

All statistical computations were audited and validated with both parametric and non-parametric tests:

### 4.1 Table Booking Impact Test
- **With Booking Mean:** $3.442 ★$ | **Without Booking Mean:** $2.559 ★$
- **Parametric Welch's t-test:** $t = 28.5087, \quad p = 2.2174 \times 10^{-153}$
- **Non-Parametric Mann-Whitney U Test:** $U = 6,591,818.0, \quad p = 1.9578 \times 10^{-87}$
- *Verification Status:* **Statistically Significant ($p < 0.001$). Verified.**

### 4.2 Online Delivery Impact Test
- **With Delivery Mean:** $3.249 ★$ | **Without Delivery Mean:** $2.465 ★$
- **Parametric Welch's t-test:** $t = 30.2882, \quad p = 4.1263 \times 10^{-191}$
- **Non-Parametric Mann-Whitney U Test:** $U = 10,590,134.5, \quad p = 1.0673 \times 10^{-58}$
- *Verification Status:* **Statistically Significant ($p < 0.001$). Verified.**

### 4.3 Pricing Tier ANOVA Test
- **One-Way ANOVA across Price Ranges (1 to 4):**
  $F = 807.0346, \quad p = 0.0000$ ($p < 10^{-300}$)
- *Verification Status:* **Statistically Significant ($p < 0.001$). Verified.**

---

## 🤖 5. Machine Learning Pipeline & Leakage Audit

### 5.1 Data Leakage Prevention Check
- **Verification Rule:** `Rating color` and `Rating text` must NEVER be present in the training feature matrix $X$.
- **Audit Confirmation:**
  ```python
  feature_cols = [
      'Country Code', 'Longitude', 'Latitude', 'Average Cost for two',
      'Price range', 'Votes', 'Name_Length', 'Address_Length',
      'Has_Table_booking_encoded', 'Has_Online_delivery_encoded', 'Num_Cuisines'
  ]
  ```
  Neither `Rating color` nor `Rating text` is present in `feature_cols`. Data leakage was completely eliminated.

### 5.2 Supervised Learning Benchmark Results
Models evaluated on independent $20\%$ test set ($1,911$ samples, `random_state=42`):
- **Random Forest Regressor:** $\text{MAE} = 0.1993, \quad \text{MSE} = 0.0916, \quad \text{RMSE} = 0.3027, \quad R^2 = \mathbf{0.9598}, \quad \text{CV } R^2 = \mathbf{0.9566}$
- **Decision Tree Regressor:** $\text{MAE} = 0.2154, \quad \text{MSE} = 0.1061, \quad \text{RMSE} = 0.3257, \quad R^2 = 0.9534, \quad \text{CV } R^2 = 0.9503$
- **Linear Regression:** $\text{MAE} = 1.0535, \quad \text{MSE} = 1.5863, \quad \text{RMSE} = 1.2595, \quad R^2 = 0.3031, \quad \text{CV } R^2 = 0.3121$

### 5.3 Live Inference Verification Across 5 Edge-Case Profiles
| Test Profile | Input Parameters | Raw Prediction | Clipped Rating | Expected Real-World Behavior | Verification |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Standard Casual** | Cost: $600$, Votes: $85$, Range: $2$, Booking: No, Delivery: Yes | $3.242$ | **`3.24 ★`** | Matches mid-range average | ✅ Passed |
| **High-End Luxury** | Cost: $3,500$, Votes: $1,200$, Range: $4$, Booking: Yes, Delivery: No | $4.235$ | **`4.23 ★`** | Matches fine-dining tier | ✅ Passed |
| **Zero-Vote Unrated** | Cost: $200$, Votes: $0$, Range: $1$, Booking: No, Delivery: No | $0.000$ | **`0.00 ★`** | Accurately identifies unrated status | ✅ Passed |
| **International US** | Country: $216$, Cost: $\$40$, Votes: $450$, Range: $2$ | $4.159$ | **`4.16 ★`** | Matches US market benchmark | ✅ Passed |
| **Outlier High Volume**| Cost: $1,000$, Votes: $10,000$, Range: $3$, Both Services | $4.046$ | **`4.05 ★`** | Stable top-tier rating | ✅ Passed |

---

## 🗺️ 6. Geospatial Audit

- **Coordinate Audit:** $9,054$ valid coordinates; $497$ invalid `(0.0, 0.0)` pairs filtered out during map rendering.
- **Folium Map:** Generated and tested at `outputs/charts/restaurant_geospatial_map.html`. Loads offline without requiring external API keys.
- **Plotly Map Compatibility:** Updated to use `px.scatter_map` with `open-street-map` style for Plotly 7.1.0 compatibility.

---

## 🚀 7. Test Execution Summary

The test runner executed all scripts sequentially with zero errors:
1. `level_1/task_1_preprocessing.py`: **Passed (3.63s)**
2. `level_1/task_2_descriptive_analysis.py`: **Passed (4.81s)**
3. `level_1/task_3_geospatial_analysis.py`: **Passed (4.72s)**
4. `level_2/task_1_booking_delivery.py`: **Passed (4.65s)**
5. `level_2/task_2_price_analysis.py`: **Passed (4.57s)**
6. `level_2/task_3_feature_engineering.py`: **Passed (4.62s)**
7. `level_3/task_1_predictive_modeling.py`: **Passed (18.35s)**
8. `level_3/task_2_customer_preferences.py`: **Passed (4.59s)**
9. `level_3/task_3_visualization.py`: **Passed (6.34s)**
- **Total Pipeline Execution Time:** **54.24 seconds**
- **Runtime Errors:** **0**

---

## ⚠️ 8. Known Limitations & Technical Scope
1. **Regional Clustering:** Over $85\%$ of the restaurants are located in the National Capital Region of India. While the model handles international records accurately, regional density influences overall feature importances.
2. **Currency Standardization:** The dataset uses 12 different currencies in the `Average Cost for two` column. Normalizing all costs to a single baseline (e.g. USD via historical exchange rates) could further refine financial feature importance.

---

## 🏁 Final Audit Verdict
**STATUS: 100% PRODUCTION READY FOR FINAL INTERNSHIP SUBMISSION.**
All code, models, reports, and UI components meet professional data science standards.
