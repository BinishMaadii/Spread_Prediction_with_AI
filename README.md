# DengAI: Predicting Disease Spread

An end-to-end Python machine learning pipeline for the DrivenData **DengAI: Predicting Disease Spread** competition. The pipeline predicts weekly dengue fever case counts for San Juan (`sj`) and Iquitos (`iq`) using climate, weather, and vegetation data.

---

## Workflow Overview

1. **Data Loading & Preprocessing:** Reads raw training and test datasets, handles missing values using linear interpolation, and aligns calendar dates across both cities.
2. **Feature Engineering:** Creates cyclical calendar encodings (`sin`/`cos` transformations) and extracts time-lagged climate indicators (temperature, humidity, vegetation index) to account for incubation delays.
3. **Model Training & Selection:** Fits three distinct model architectures per city on historical training data:
   * **Negative Binomial GLM:** Generalized Linear Model suited for over-dispersed count data.
   * **Random Forest Regressor:** Handles complex non-linear feature interactions.
   * **LightGBM Regressor:** Gradient boosting configured with a Poisson objective for skewed count distribution.
4. **Validation:** Evaluates models on a held-out temporal validation set (the final 20% of training weeks) using Mean Absolute Error (MAE) and selects the best-performing model for each city independently.
5. **Inference & Submission:** Refits winning models on the full dataset, generates predictions for the competition test set, clips negative values.

---

## Directory Setup

Place the competition data files in a `data/` folder in the same directory as the script:

```text
.
├── dengai_beginner.py
├── README.md
├── data/
│   ├── dengue_features_train.csv
│   ├── dengue_labels_train.csv
│   └── dengue_features_test.csv
├── plots/                # Auto-generated visualization outputs
