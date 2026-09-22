import os
import json
import warnings
from datetime import datetime
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
import lightgbm as lgb
 
# statsmodels prints a lot of convergence warnings while we try different alphas
warnings.filterwarnings("ignore")
 
# =====================================================================
# SETTINGS
# =====================================================================
DATA_DIR = "/kaggle/input/datasets/binishbatool/dengue-time-series"  # or "data"
PLOTS_DIR = "plots"
MODELS_DIR = "models"
RANDOM_STATE = 42
CITIES = ["sj", "iq"]
 
# Walk-forward validation: 4 folds, each one year (52 weeks) long.
# The last 4 years of each city's training data are used as validation years.
N_FOLDS = 4
HORIZON = 52
 
# Values of the Negative Binomial alpha to try when tuning
ALPHA_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
 
# The original settings from dengai_beginner.py
BEFORE = {
    "name": "before",
    "rows_per_city": 250,          # only the most recent 250 weeks are used for training
    "fill_method": "interpolate",  # straight line between the weeks before AND after a gap
    "lags": [1, 2, 4],
    "windows": [4],
    "negbin_alpha": 1.0,
    "rf_trees": 150,
    "rf_depth": 6,
    "lgb_rounds": 200,
    "lgb_leaves": 15,
    "lgb_learning_rate": 0.05,
    "use_ensemble": False,
}
 

AFTER = {
    "name": "after",
    "rows_per_city": None,         # None = use all available history
    "fill_method": "past_only",    # copy the last known value forward
    "lags": [1, 2, 3, 4, 8],
    "windows": [4, 8],
    "negbin_alpha": "tune",        # pick the best alpha from ALPHA_GRID
    "rf_trees": 400,
    "rf_depth": 8,
    "lgb_rounds": 400,
    "lgb_leaves": 31,
    "lgb_learning_rate": 0.03,
    "use_ensemble": True,
}
 
ID_COLUMNS = ["city", "year", "weekofyear", "week_start_date", "total_cases"]
 
CLIMATE_COLUMNS = [
    "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw",
    "precipitation_amt_mm",
    "reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k",
    "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k",
    "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent",
    "reanalysis_sat_precip_amt_mm", "reanalysis_specific_humidity_g_per_kg",
    "reanalysis_tdtr_k",
    "station_avg_temp_c", "station_diur_temp_rng_c",
    "station_max_temp_c", "station_min_temp_c", "station_precip_mm",
]
 
# The GLM only uses these 4 columns (lag 2 and window 4 exist in both setups)
GLM_COLUMNS = [
    "reanalysis_specific_humidity_g_per_kg_lag2",
    "station_avg_temp_c_lag2",
    "reanalysis_specific_humidity_g_per_kg_avg4wk",
    "station_avg_temp_c_avg4wk",
]
 
# Models we are allowed to pick as the final model (baselines are only yardsticks)
REAL_MODELS = ["negbin_glm", "random_forest", "lightgbm", "ensemble"]
SINGLE_MODELS = ["negbin_glm", "random_forest", "lightgbm"]
 
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
 
plot_number = [0]

 
def save_plot(fig, name):
    plot_number[0] += 1
    path = os.path.join(PLOTS_DIR, f"{plot_number[0]:02d}_{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    #plt.close(fig)
    
    #plt.show()
    
    display(fig) # in order to render the plot on screen in cell
    print(f"  saved plot -> {path}")
 



# =====================================================================
# STEP 1: Load the raw data. This data is labeled
# =====================================================================
print("\n=== STEP 1: Loading data ===")
features_train = pd.read_csv(os.path.join(DATA_DIR, "dengue_features_train.csv"),
                             parse_dates=["week_start_date"])
labels_train = pd.read_csv(os.path.join(DATA_DIR, "dengue_labels_train.csv"))
features_test = pd.read_csv(os.path.join(DATA_DIR, "dengue_features_test.csv"),
                            parse_dates=["week_start_date"])
 
features_train = features_train.sort_values(["city", "week_start_date"]).reset_index(drop=True)
features_test = features_test.sort_values(["city", "week_start_date"]).reset_index(drop=True)
train_df = features_train.merge(labels_train, on=["city", "year", "weekofyear"], how="left")
print("Training rows:", len(train_df), " Test rows:", len(features_test))
 

