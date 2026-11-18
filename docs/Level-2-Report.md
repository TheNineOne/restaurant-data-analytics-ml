# 🛎️ Level 2: Business and Feature Analysis Report
### Cognifyz Technologies Data Science Internship Project
**Project Name:** Restaurant Data Analytics & Rating Prediction System  
**Prepared by:** Senior Data Scientist & QA Lead  
**Dataset Reference:** `Dataset .csv` (9,551 records, 21 columns)

---

## 📌 Executive Summary
Level 2 transitions from macro-level exploration into operational hospitality analytics. We evaluate the direct commercial and perceptual impact of convenience services (table booking and online delivery), analyze pricing elasticity across tiers, examine rating color calibrations, and engineer domain features that prepare the dataset for predictive machine learning.

---

## 🍽️ Task 1: Table Booking and Online Delivery Dynamics

### 1.1 Service Adoption Rates
- **Table Booking Availability:**
  - **Offers Table Booking (Yes):** `1,158` restaurants (**`12.12%`**)
  - **No Table Booking (No):** `8,393` restaurants (**`87.88%`**)
- **Online Delivery Availability:**
  - **Offers Online Delivery (Yes):** `2,451` restaurants (**`25.66%`**)
  - **No Online Delivery (No):** `7,100` restaurants (**`74.34%`**)

### 1.2 Rating Impact Analysis & Statistical Hypothesis Testing
Does offering table booking or online delivery produce a statistically significant improvement in customer rating?

#### (A) Table Booking Rating Test
- **Average Rating WITH Table Booking:** **`3.442 ★`** (Median: $3.60 ★$)
- **Average Rating WITHOUT Table Booking:** **`2.559 ★`** (Median: $3.10 ★$)
- **Perception Premium (Delta):** **`+0.883 ★`**
- **Welch's Two-Sample t-test:** $t = 28.5087$, $p\text{-value} = 2.2174 \times 10^{-153}$
- **Mann-Whitney U Non-Parametric Test:** $U = 6,591,818.0$, $p\text{-value} = 1.9578 \times 10^{-87}$
- *Statistical Conclusion:* The null hypothesis ($H_0$) of equal mean ratings is rejected. Venues offering table booking maintain a statistically significant rating advantage.

#### (B) Online Delivery Rating Test
- **Average Rating WITH Online Delivery:** **`3.249 ★`** (Median: $3.40 ★$)
- **Average Rating WITHOUT Online Delivery:** **`2.465 ★`** (Median: $3.10 ★$)
- **Perception Premium (Delta):** **`+0.784 ★`**
- **Welch's Two-Sample t-test:** $t = 30.2882$, $p\text{-value} = 4.1263 \times 10^{-191}$
- **Mann-Whitney U Test:** $U = 10,590,134.5$, $p\text{-value} = 1.0673 \times 10^{-58}$
- *Statistical Conclusion:* Online delivery accessibility strongly correlates with higher review scores.

### 1.3 Online Delivery Availability Across Price Tiers
Cross-tabulation reveals a distinct non-linear operational dynamic:

| Price Range | Tier Label | Total Restaurants | Online Delivery (No) | Online Delivery (Yes) | Delivery Adoption % |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Budget | 4,444 | 3,743 | 701 | **15.77%** |
| **2** | Mid-Range | 3,113 | 1,827 | 1,286 | **41.31%** |
| **3** | Expensive | 1,408 | 997 | 411 | **29.19%** |
| **4** | Luxury / Fine Dining | 586 | 533 | 53 | **9.04%** |

#### Operational Insights:
1. **The Mid-Range Sweet Spot:** Price Range 2 exhibits the highest delivery penetration ($41.31\%$), acting as the primary workhorse for food delivery aggregators.
2. **The Fine Dining Paradox:** Luxury restaurants (Price Range 4) show the lowest delivery adoption ($9.04\%$). Luxury brands consciously restrict off-premises consumption to protect plating quality, service atmosphere, and prestige.

---

## 💰 Task 2: Price Range Analysis

### 2.1 Price Range Frequency Breakdown
- **Price Range 1 (Budget):** `4,444` venues (**`46.53%`**) — *Most Common Price Range*
- **Price Range 2 (Mid-Range):** `3,113` venues (**`32.59%`**)
- **Price Range 3 (Expensive):** `1,408` venues (**`14.74%`**)
- **Price Range 4 (Luxury):** `586` venues (**`6.14%`**)

### 2.2 Rating Statistics by Price Range
| Price Range | Count | Mean Rating | Median | Std Dev | Min | Max |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4,444 | **1.9999** | 2.90 | 1.6124 | 0.0 | 4.9 |
| **2** | 3,113 | **2.9411** | 3.30 | 1.2374 | 0.0 | 4.9 |
| **3** | 1,408 | **3.6834** | 3.80 | 0.7910 | 0.0 | 4.9 |
| **4** | 586 | **3.8179** | 3.90 | 0.7398 | 0.0 | 4.9 |

- **One-Way ANOVA Test:** $F = 807.0346$, $p\text{-value} = 0.0000$ ($p < 10^{-300}$).
- *Key Takeaway:* There is a strictly monotonic, statistically proven relationship between pricing tier and customer rating. As restaurants move up the price spectrum from Range 1 to 4, mean ratings climb by **`+1.818 ★`**.

### 2.3 Rating Color Calibration
| Rating Color | Count | Mean Rating | Median Rating | Standard Quality Category |
| :--- | :---: | :---: | :---: | :--- |
| **Dark Green** | 301 | **4.6598 ★** | 4.60 | Top Tier / Exceptional ($\ge 4.5 ★$) |
| **Green** | 1,079 | **4.1681 ★** | 4.10 | High Quality / Very Good ($4.0 - 4.4 ★$) |
| **Yellow** | 2,100 | **3.6834 ★** | 3.70 | Solid Performance / Good ($3.5 - 3.9 ★$) |
| **Orange** | 3,737 | **3.0516 ★** | 3.10 | Acceptable / Average ($2.5 - 3.4 ★$) |
| **Red** | 186 | **2.2978 ★** | 2.30 | Subpar / Poor ($1.8 - 2.4 ★$) |
| **White** | 2,148 | **0.0000 ★** | 0.00 | Unrated / Newly Listed |

- **Color with Highest Average Rating:** **`Dark Green`** ($4.6598 ★$).

---

## 🛠️ Task 3: Feature Engineering

### 3.1 Derived Features & Rationale
1. **`Name_Length`:** Length of the restaurant name in characters.
   - *Min:* $2$, *Max:* $54$, *Mean:* $15.16 \pm 6.44$.
2. **`Address_Length`:** Length of the physical restaurant address string.
   - *Min:* $13$, *Max:* $132$, *Mean:* $53.54 \pm 22.84$.
3. **`Has_Table_booking_encoded`:** Binary integer encoding (`1` for Yes, `0` for No).
4. **`Has_Online_delivery_encoded`:** Binary integer encoding (`1` for Yes, `0` for No).
5. **`Is_Delivering_now_encoded`:** Binary integer encoding (`1` for Yes, `0` for No).
6. **`Num_Cuisines`:** Count of distinct cuisines offered by parsing the comma-separated string.
   - *Min:* $0$, *Max:* $8$, *Mean:* $2.06 \pm 1.13$.
7. **`Country_Name`:** Human-readable country mapping from ISO numeric codes.
8. **`Price_Category`:** Categorical tier description (Budget, Mid-Range, Expensive, Luxury).

### 3.2 Correlation with Target (`Aggregate rating`)
| Feature | Pearson Correlation ($r$) | Direction & Significance |
| :--- | :---: | :--- |
| **Price range** | **+0.4379** | Moderate positive linear relationship |
| **Votes** | **+0.3137** | Moderate positive relationship |
| **Num_Cuisines** | **+0.2484** | Weak positive relationship |
| **Has_Online_delivery_encoded** | **+0.2257** | Weak positive relationship |
| **Has_Table_booking_encoded** | **+0.1900** | Weak positive relationship |
| **Average Cost for two** | **+0.0518** | Very weak positive relationship |
| **Address_Length** | **+0.0023** | Negligible linear correlation |
| **Name_Length** | **-0.0352** | Negligible linear correlation |

### 3.3 Dataset Export
- Processed, type-validated dataset saved to: `data/processed/cleaned_restaurant_data.csv`.
- Full engineered dataset with all 8 new attributes (29 total columns) saved to: `data/processed/engineered_restaurant_data.csv`.
