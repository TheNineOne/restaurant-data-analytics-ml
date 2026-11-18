# Dataset Architecture & Local Setup

## Overview
This repository uses the restaurant dataset provided for the Cognifyz Technologies Data Science Internship Program.

## Data Files
- `raw/Dataset .csv`: Raw restaurant records (9,551 rows, 21 columns).
- `processed/cleaned_restaurant_data.csv`: Validated dataset with imputed missing values and standardized data types.
- `processed/engineered_restaurant_data.csv`: Enhanced dataset featuring 8 domain-specific engineered attributes (29 columns total).

## Schema Summary
- **Target Variable:** `Aggregate rating` (Continuous, range: 0.0 to 4.9)
- **Key Features:** `Country Code`, `City`, `Cuisines`, `Average Cost for two`, `Price range`, `Votes`, `Has Table booking`, `Has Online delivery`
- **Engineered Features:** `Name_Length`, `Address_Length`, `Has_Table_booking_encoded`, `Has_Online_delivery_encoded`, `Num_Cuisines`, `Country_Name`, `Price_Category`

## Reproducibility
To regenerate processed datasets from the raw CSV, execute:
```bash
python level_2/task_3_feature_engineering.py
```
