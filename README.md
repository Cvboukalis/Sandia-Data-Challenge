# [2025 Sandia Data Challenge](https://github.com/Cvboukalis/Sandia-Data-Challenge/blob/main/Presentation.pdf)
This repo contains our project for the Sandia Data Challenge 2025, where our team built various machine learning models to reduce the scrap rate for 3D-printed metal parts produced with Laser Powder Bed Fusion (LPBF).
## Project Overview & Goal

For this project, we were given a dataset comprised of different components for stainless-steel cups built on metal plates. Essentially, each part has to meet specific dimensional requirements (inner diameter, outer diameter, height) to be accepted; otherwise it's considered "scrapped". There were several different main parameters:

- Build plate layout (e.g., 6×6, 6×6 with test artifacts, 11×11 with test artifacts) 
- Location of each unit on the build plate  
- Powder type (virgin vs recycled)
  
## Business Question

> **Which combinations of build parameters minimize the probability of scrap, and by how much can we reduce the scrap rate from the original data?** 

## Modeling
We saw this problem as a **binary classification task** to predict whether a part would be scrapped based on:

- Build plate layout  
- Zone / location on the plate
- Powder type 

After cleaning and feature engineering, we trained and compared two main models:

### 1. [Random Forest](https://github.com/Cvboukalis/Sandia-Data-Challenge/blob/main/RandomForest.ipynb)
### 2. XGBoost
## Results

We compared the baseline original scrap rate to the best predicted combination of features that each model chose when we searched over combinations of layout, zone, and powder type.

### XGBoost (Best Model)

- **Original Scrap Rate:** `0.21153846153846154`  (~21.2%)  
- **Best Predicted Scrap Rate:** `0.056874078`  (~5.7%)  
- **Reduction:** ~**15.5 percentage points** in projected scrap rate
  
The best XGBoost model projects that scrapped parts could be reduced from roughly **1 in 5 parts** to about **1 in 18 parts**.

### Random Forest

- **Original Scrap Rate:** `0.21153846153846154`  (~21.2%)  
- **Best Predicted Scrap Rate:** `0.085382`  (~8.5%)  
- **Reduction:** ~**12.6 percentage points**
  
We also found similar trends with the Random Forest model (better layouts, better zones, and virgin powder), but its best predicted scrap rate wasn't as low as XGBoost. 
