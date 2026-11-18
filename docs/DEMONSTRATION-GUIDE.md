# 🎬 Mentor Demonstration Guide & LinkedIn Video Script
### Cognifyz Technologies Data Science Internship Project
**Project Name:** Restaurant Data Analytics & Rating Prediction System  
**Internship Role:** Senior Data Scientist & ML Engineer  
**Target Duration for Video:** 3 to 5 Minutes (Meets Cognifyz Submission Criteria)

---

## 📌 Part 1: Cognifyz Mentor Evaluation Guide

### 1. Project Introduction & Purpose
"Good morning / afternoon, respected mentors. I am presenting my end-to-end Data Science Internship Project completed with Cognifyz Technologies: the **Restaurant Data Analytics & Rating Prediction System**.

The primary business objective of this project is twofold:
1. Conduct exploratory, descriptive, and geospatial analytics on a global dataset of **9,551 restaurants** across 15 countries to identify the key operational and financial drivers of customer satisfaction.
2. Build, evaluate, and deploy a supervised machine learning regression pipeline to accurately predict restaurant aggregate ratings without target leakage, making it accessible through an interactive Streamlit intelligence portal."

### 2. Dataset Overview
- **Source:** Real-world Zomato multi-country restaurant dataset.
- **Scale:** 9,551 restaurants, 21 initial columns, 141 unique cities across 15 countries.
- **Target Variable:** `Aggregate rating` (Continuous float, 0.0 to 4.9).
- **Core Findings on Target Skew:** Exactly 2,148 venues (22.49%) have a rating of 0.0 ("Not rated"). Active rated restaurants average 3.44 ★ with a Gaussian distribution.

### 3. Machine Learning Approach & Prevention of Data Leakage
- **Leakage Prevention:** `Rating text` and `Rating color` were strictly omitted from training features because they directly expose the target rating. `Restaurant ID` was also dropped as a non-predictive surrogate key.
- **Features Used (11 Features):** Location (`Country Code`, `Longitude`, `Latitude`), Economics (`Average Cost for two`, `Price range`), Engagement (`Votes`), Operations (`Has_Table_booking_encoded`, `Has_Online_delivery_encoded`, `Num_Cuisines`), and Text Engineering (`Name_Length`, `Address_Length`).
- **Benchmark Comparison:**
  - *Linear Regression:* $R^2 = 0.3031$, $\text{MAE} = 1.0535$ (Underfits non-linear interaction terms).
  - *Decision Tree:* $R^2 = 0.9534$, $\text{MAE} = 0.2154$.
  - *Random Forest Regressor:* **$R^2 = 0.9598$, $\text{MAE} = 0.1993$, $\text{RMSE} = 0.3027$** (Explains 95.98% of variance; serialized to `models/best_model.joblib`).
  - *Feature Drivers:* Customer vote count ($96.57\%$) is the single most critical anchor for rating stability.

---

## 🎥 Part 2: LinkedIn Screen Recording Script (3 to 5 Minutes)

*Hashtags to include in LinkedIn post:* `#cognifyz #cognifyzTech #cognifyzTechnologies #DataScience #MachineLearning #Streamlit #Python`

---

### **[0:00 - 0:45] Scene 1: Introduction & Executive Overview**
- **Action on Screen:** Open the Streamlit Dashboard at `http://localhost:8501`. Have the sidebar showing "1. Overview".
- **Spoken Script:**
  > *"Hello everyone! I am thrilled to showcase my final internship project for Cognifyz Technologies: the **Restaurant Data Analytics & Rating Prediction System**.
  > 
  > Here on the executive overview, our system processes over 9,550 restaurants across 15 countries. You can see our 5 live KPI cards tracking total restaurants, urban cities, global countries, average rating, and over 1.49 million customer votes. 
  > 
  > Notice the dynamic sidebar filters that allow stakeholders to slice data by country, city, price range, and rating threshold in real time, with the ability to download filtered datasets on demand."*

---

### **[0:45 - 1:30] Scene 2: Data Exploration & Geospatial Intelligence**
- **Action on Screen:** Click on **"2. Data Exploration"** in the sidebar, scroll through the missing value audit and distribution charts. Then switch to **"4. Geospatial Analysis"** and toggle the interactive map.
- **Spoken Script:**
  > *"In Level 1, we conducted thorough exploratory data analysis. We discovered that only 9 rows in Cuisines were missing, which we imputed cleanly. We also identified a major class imbalance: 22.5% of venues carry a 0.0 rating because they are newly listed or unrated, while actively rated venues average 3.44 stars.
  > 
  > Moving to our Geospatial Analysis page, we audited geographic coordinates. We isolated and filtered 497 records with 0,0 coordinates in Delhi NCR to prevent ocean clustering, and rendered an interactive map that visualizes global dining density sized by customer votes and color-coded by rating tier."*

---

### **[1:30 - 2:30] Scene 3: Business Dynamics & Price Range Analysis**
- **Action on Screen:** Click on **"5. Booking & Delivery Analysis"**, highlight the boxplots. Then switch to **"6. Price Range Analysis"**.
- **Spoken Script:**
  > *"In Level 2, we evaluated service dynamics. Our statistical tests revealed a massive insight: restaurants offering **Table Booking** score an average rating of **3.44 ★** compared to **2.56 ★** without—a statistically significant rating premium of nearly a full star ($t = 28.51, p < 0.001$).
  > 
  > On the Price Range Analysis page, we observed a strict monotonic relationship: as venues climb from Budget Tier 1 to Luxury Tier 4, average ratings rise from 2.00 to 3.82 stars. Furthermore, Mid-Range Tier 2 restaurants capture peak online delivery adoption at 41.3%, whereas Luxury fine-dining venues restrict online delivery to just 9% to protect the in-house guest experience."*

---

### **[2:30 - 3:30] Scene 4: Machine Learning Benchmark**
- **Action on Screen:** Click on **"8. Machine Learning"**. Scroll through the model comparison table, R² bar chart, and feature importance chart.
- **Spoken Script:**
  > *"In Level 3, we built our supervised machine learning regression engine. To maintain academic rigor, we strictly prevented data leakage by omitting 'Rating text' and 'Rating color', as they directly reveal the target rating.
  > 
  > We benchmarked Linear Regression, a Decision Tree, and a Random Forest Regressor using an 80/20 train-test split and 5-fold cross-validation. 
  > 
  > Our champion model is the **Random Forest Regressor**, achieving an **R² score of 0.9598** and a **Mean Absolute Error of only 0.199 rating points**. Review volume and location were identified as the strongest predictors."*

---

### **[3:30 - 4:45] Scene 5: Live Real-Time Rating Prediction Demo**
- **Action on Screen:** Click on **"9. Model Prediction"**. Enter a sample restaurant profile:
  - *Name:* "Spice Garden Fine Dining"
  - *Address:* "Sector 29, City Center, Gurgaon"
  - *Country:* India
  - *Cost for two:* 1200
  - *Price Range:* 3
  - *Votes:* 350
  - *Table Booking:* Yes
  - *Online Delivery:* Yes
  - *Cuisines:* North Indian, Mughlai, Continental
  - Click the **"⚡ Predict Aggregate Rating"** button. Show the predicted score card (e.g. ~3.9 - 4.1 ★).
- **Spoken Script:**
  > *"Finally, here is our live Model Prediction module. Stakeholders can input a restaurant's profile—including name, location, price range, review volume, and convenience amenities.
  > 
  > When I click 'Predict Aggregate Rating', our serialized Random Forest pipeline extracts engineered features on the fly and predicts an expected rating of **4.0 ★** with an automated quality badge of 'Very Good'. 
  > 
  > The system also displays explanation breakdown notes on why this rating was assigned based on review momentum and service synergy."*

---

### **[4:45 - 5:00] Scene 6: Conclusion & Thank You**
- **Action on Screen:** Return to the Overview page.
- **Spoken Script:**
  > *"This concludes my project demonstration. I would like to express my sincere gratitude to the team and mentors at **Cognifyz Technologies** for this rewarding internship experience. 
  > 
  > All source code, serialized models, reports, and Jupyter Notebooks are available in my repository. Thank you!"*

---

## 📌 Part 3: Ready-to-Copy LinkedIn Post Template

```text
🚀 Excited to share the completion of my Data Science Internship Project with Cognifyz Technologies! 🍽️📊

Over the course of this internship, I built the "Restaurant Data Analytics & Rating Prediction System", an end-to-end data science and machine learning solution analyzing 9,551 restaurants across 15 countries.

🔍 Key Project Highlights:
✅ Data Exploration & Imbalance: Audited 9,551 records; diagnosed and handled target skewness where 22.5% of venues were unrated.
✅ Geospatial Intelligence: Filtered unmapped coordinates and built an interactive Folium clustered map.
✅ Business Dynamics: Statistically proved that offering Table Booking delivers a +0.88 star rating premium (p < 0.001 via Welch's t-test).
✅ Machine Learning Engine: Trained Linear Regression, Decision Trees, and Random Forest Regressors without data leakage. The champion Random Forest model achieved an impressive R² of 0.9598 and MAE of 0.199!
✅ Enterprise Web Dashboard: Deployed a 9-page interactive Streamlit dashboard featuring live real-time rating predictions.

Check out the 4-minute demonstration video below! 🎥👇

A huge thank you to Cognifyz Technologies for providing this hands-on industry opportunity!

#cognifyz #cognifyzTech #cognifyzTechnologies #DataScience #MachineLearning #Python #Streamlit #DataAnalytics #AI
```
