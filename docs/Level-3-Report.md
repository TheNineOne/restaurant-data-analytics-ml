# 🤖 Level 3: Machine Learning and Visualization Report
### Cognifyz Technologies Data Science Internship Project
**Project Name:** Restaurant Data Analytics & Rating Prediction System  
**Prepared by:** Senior Data Scientist & QA Lead  
**Dataset Reference:** `Dataset .csv` (9,551 records, 21 columns)

---

## 📌 Executive Summary
Level 3 encapsulates the core predictive intelligence and visual synthesis of the internship project. We developed, benchmarked, and serialized supervised machine learning regressors to predict continuous restaurant ratings while strictly enforcing data leakage prevention. In parallel, customer preferences were investigated to discover the volume-versus-satisfaction paradox across 145 global cuisines, culminating in a 6-panel master visual analytics dashboard.

---

## 🎯 Task 1: Predictive Modeling & Performance Benchmark

### 1.1 Data Leakage Prevention Guardrail
In rigorous data science workflows, target leakage compromises model validity.
- **Excluded Features:**
  - `Rating text` (`"Excellent"`, `"Good"`, `"Average"`, etc.) and `Rating color` (`"Dark Green"`, `"Red"`, etc.) were **strictly excluded**. Both features are direct categorizations created after the aggregate rating is established; including them would introduce catastrophic target leakage and create an artificial 100% accurate model that fails on new data.
  - `Restaurant ID` was excluded as an arbitrary identifier.
- **Included Feature Vector (11 Features):**
  `Country Code`, `Longitude`, `Latitude`, `Average Cost for two`, `Price range`, `Votes`, `Name_Length`, `Address_Length`, `Has_Table_booking_encoded`, `Has_Online_delivery_encoded`, `Num_Cuisines`.

### 1.2 Training Methodology
- **Train/Test Split:** $80\%$ Training ($7,640$ observations) / $20\%$ Testing ($1,911$ observations).
- **Random Seed:** Fixed at `random_state=42` to guarantee exact mathematical reproducibility.
- **Cross-Validation:** 5-Fold Stratified K-Fold Cross Validation evaluated across the training partition.

### 1.3 Supervised Regression Benchmark
Three distinct algorithm families were trained and benchmarked using four standard regression metrics:
$$\text{MAE} = \frac{1}{n} \sum |y - \hat{y}|, \quad \text{MSE} = \frac{1}{n} \sum (y - \hat{y})^2, \quad \text{RMSE} = \sqrt{\text{MSE}}, \quad R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$$

| Algorithm | Train $R^2$ | Test MAE | Test MSE | Test RMSE | Test $R^2$ Score | 5-Fold CV $R^2$ | Model Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.9757** | **0.1993** | **0.0916** | **0.3027** | **0.9598** | **0.9566** | 🏆 **BEST (Production)** |
| **Decision Tree Regressor** | 0.9596 | 0.2154 | 0.1061 | 0.3257 | 0.9534 | 0.9503 | Competitive Non-Linear |
| **Linear Regression** | 0.3149 | 1.0535 | 1.5863 | 1.2595 | 0.3031 | 0.3121 | Underfitting Baseline |

### 1.4 Model Selection & Discussion
- **Linear Regression Underfitting:** Linear Regression explains only $30.31\%$ of the variance with an RMSE of $1.2595$ rating points. The relationship between restaurant characteristics and consumer ratings is non-linear and dominated by threshold interactions (e.g. review volume thresholds).
- **Decision Tree Robustness:** The tuned Decision Tree achieved an $R^2$ of $0.9534$, capturing non-linear feature splits effectively.
- **Random Forest Superiority:** Random Forest Regressor achieved the highest predictive accuracy:
  - **$R^2 = 0.9598$:** Explains **$95.98\%$** of rating variance on unseen test data.
  - **$\text{MAE} = 0.1993$:** Predictions deviate from actual ratings by less than **$0.20$ stars** on average.
  - **$5$-Fold CV $R^2 = 0.9566$:** Confirms strong generalization with minimal variance across folds.

### 1.5 Feature Importance Ranking (Random Forest)
| Feature Name | Relative Importance | Percentage Contribution | Analytical Significance |
| :--- | :---: | :---: | :--- |
| **Votes** | **0.9657** | **96.57%** | Review count is the single most decisive anchor for rating stability. |
| **Longitude** | 0.0113 | 1.13% | Captures regional market dynamics and urban concentration. |
| **Latitude** | 0.0079 | 0.79% | Captures localized geographic micro-clusters. |
| **Average Cost for two** | 0.0040 | 0.40% | Financial tier indicator. |
| **Country Code** | 0.0035 | 0.35% | Macro-economic dining market differentiation. |
| **Address_Length** | 0.0032 | 0.32% | Specificity and quality of address description. |
| **Name_Length** | 0.0024 | 0.24% | Brand naming complexity. |
| **Num_Cuisines** | 0.0009 | 0.09% | Menu diversification indicator. |
| **Price range** | 0.0005 | 0.05% | Pricing category. |
| **Has_Online_delivery_encoded** | 0.0004 | 0.04% | Delivery service availability. |
| **Has_Table_booking_encoded** | 0.0003 | 0.03% | Reservation service availability. |

### 1.6 Model Serialization & Deployment
- The champion Random Forest model was serialized to [`models/best_model.joblib`](file:///d:/nineone/study/interships/Cognifyz%20IT%20Solutions%2024%20nov%20%20(Last%20Date%20of%20Task%20Submission)/restaurant-data-science/models/best_model.joblib).
- Comprehensive metadata including hyperparameters, training split shapes, and evaluation metrics was exported to [`models/model_metadata.json`](file:///d:/nineone/study/interships/Cognifyz%20IT%20Solutions%2024%20nov%20%20(Last%20Date%20of%20Task%20Submission)/restaurant-data-science/models/model_metadata.json).

---

## 🍜 Task 2: Customer Preference Analysis

### 2.1 The Volume vs Quality Paradox
Expanding the comma-separated cuisines into 19,710 individual restaurant-cuisine associations reveals a classic data science phenomenon:

#### Top 10 Most Popular Cuisines (Ranked by Customer Votes)
| Cuisine | Restaurant Count | Total Customer Votes | Average Rating |
| :--- | :---: | :---: | :---: |
| **North Indian** | 3,960 | **595,981** | 2.5105 ★ |
| **Chinese** | 2,735 | **364,351** | 2.6210 ★ |
| **Italian** | 764 | **329,265** | 3.5620 ★ |
| **Continental** | 736 | **288,255** | 3.5188 ★ |
| **Fast Food** | 1,986 | **184,058** | 2.5631 ★ |
| **American** | 390 | **183,117** | 3.6615 ★ |
| **Cafe** | 703 | **177,568** | 3.3212 ★ |
| **Mughlai** | 995 | **151,946** | 2.6100 ★ |
| **Desserts** | 653 | **105,889** | 2.9789 ★ |
| **Asian** | 233 | **104,303** | 3.8150 ★ |

#### Top 10 Highest Rated Cuisines (Minimum 20 Restaurants Sample Threshold)
To eliminate sample size bias (e.g. a single boutique restaurant scoring 4.9 skewing results), a 20-venue threshold was applied:
| Cuisine | Restaurant Count | Total Votes | Average Rating |
| :--- | :---: | :---: | :---: |
| **International** | 21 | 8,122 | **4.2476 ★** |
| **Southern** | 24 | 13,939 | **4.1292 ★** |
| **Vegetarian** | 23 | 10,714 | **4.0739 ★** |
| **Sandwich** | 53 | 23,500 | **4.0660 ★** |
| **Grill** | 21 | 4,301 | **4.0571 ★** |
| **Steak** | 62 | 25,677 | **3.9855 ★** |
| **Sushi** | 75 | 20,582 | **3.9733 ★** |
| **Goan** | 20 | 11,488 | **3.9700 ★** |
| **Breakfast** | 41 | 16,097 | **3.9659 ★** |
| **Mediterranean** | 112 | 80,538 | **3.9482 ★** |

*Strategic Business Insight:* Mass-market staples (North Indian, Chinese, Fast Food) dominate consumer engagement volume but suffer from lower average ratings due to high saturation of unrated and mediocre neighborhood eateries. In contrast, specialized culinary formats (Steak, Sushi, Mediterranean, International) command high customer satisfaction ($>3.95 ★$) and represent high-margin expansion opportunities.

---

## 📊 Task 3: Comprehensive Data Visualization

### 3.1 Visual Analytics Deliverables
1. **Master 6-Panel Visual Grid:** [`outputs/charts/level_3_task_3_comprehensive_dashboard.png`](file:///d:/nineone/study/interships/Cognifyz%20IT%20Solutions%2024%20nov%20%20(Last%20Date%20of%20Task%20Submission)/restaurant-data-science/outputs/charts/level_3_task_3_comprehensive_dashboard.png)
   - *Panel 1:* Aggregate Rating Histogram with Kernel Density Estimate.
   - *Panel 2:* Top 10 Cities Presence Bar Chart.
   - *Panel 3:* Price Range vs Rating Violin/Quartile Plot.
   - *Panel 4:* Customer Votes vs Rating Log-linear Regression Trend.
   - *Panel 5:* Service Combination Matrix (Booking $\times$ Delivery).
   - *Panel 6:* Full Feature Correlation Matrix.
2. **Standalone Lower-Triangle Heatmap:** [`outputs/charts/level_3_task_3_correlation_heatmap.png`](file:///d:/nineone/study/interships/Cognifyz%20IT%20Solutions%2024%20nov%20%20(Last%20Date%20of%20Task%20Submission)/restaurant-data-science/outputs/charts/level_3_task_3_correlation_heatmap.png)
3. **Model Evaluation Charts:**
   - Benchmark bar chart: `level_3_task_1_model_comparison.png`
   - Actual vs Predicted scatter: `level_3_task_1_actual_vs_predicted.png`
   - Cuisine preferences bar chart: `level_3_task_2_cuisine_preferences.png`
