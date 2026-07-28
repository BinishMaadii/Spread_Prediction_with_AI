
"""
python dengai_beginner.py

Needs these files in a "data" folder next to this script (download them
from the competition

n's Data tab; you need a free DrivenData account):
dat a /dengue_features_train.csv
dat a /dengue_labels_train.csv
dat a /dengue_features_test.csv

It will create a "plots" folder full of PNG images, and a "submission.csv"
file with your final predictions.

Python packages needed:
    pip install pandas numpy matplotlib sciki t -learn statsmodels lightgbm
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
import lightgbm as lgb

# =====================================================================
# SPEED KNOBS -- change these numbers to trade off SPEED vs ACCURACY
# =====================================================================
# While you're still learning / debugging the script, keep these small so
# each run only takes a few seconds. Once you understand what the script
# does and want the best possible accuracy, follow the "for best accuracy"
# notes next to each setting.

# How many of the most recent training weeks to use, per city.
# - Smaller number  = faster runs, but less data for the model to learn from.
# - Set to None     = use ALL available training data (936 weeks for San
#                      Juan, 520 for Iquitos) for best accuracy. This is
#                      still small data by machine-learning standards, so
#                      even "None" trains in well under a minute.
ROWS_PER_CITY = 250  # <-- try None once you're ready for a full run

# How many past weeks we "look back" to build lag features (see STEP 5 for
# what a lag feature is).
# - Fewer lags   = fewer columns = faster.
# - For best accuracy, try LAG_WEEKS = [1, 2, 3, 4, 8]
LAG_WEEKS = [1, 2, 4]

# How many weeks we average over for the rolling-average features.
# - For best accuracy, try ROLLING_WINDOWS = [4, 8]
ROLLING_WINDOWS = [4]

# Random Forest settings (see STEP 7). More trees = slower but usually a
# little more accurate, with diminishing returns past a few hundred.
RF_N_ESTIMATORS = 150  # <-- for best accuracy, try 400-600
RF_MAX_DEPTH = 6  # <-- for best accuracy, try tuning between 4 and 10

# LightGBM settings (see STEP 7). More rounds = slower but usually a little
# more accurate, with diminishing returns past a few hundred.
LGB_N_ESTIMATORS = 200  # <-- for best accuracy, try 400-800
LGB_NUM_LEAVES = 15  # <-- for best accuracy, try 31 (LightGBM's default)
LGB_LEARNING_RATE = 0.05  # <-- smaller learning rate + more rounds usually improves accuracy

# Where things get saved.
DATA_DIR = "data"
PLOTS_DIR = "plots"

RANDOM_STATE = 42
CITIES = ["sj", "iq"]

os.makedirs(PLOTS_DIR, exist_ok=True)
plot_number = [0]  # a little counter so plot filenames sort in the order they were made


def save_plot(fig, name):
    """Save a matplotlib figure into the plots/ folder with a numbered
    filename, e.g. plots/03_missing_values.png, so they're easy to browse
    in the order the script produces them."""
    plot_number[0] += 1
    path = os.path.join(PLOTS_DIR, f"{plot_number[0]:02d}_{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {path}")


# =====================================================================
# STEP 1: Load the raw data
# =====================================================================
print("\n=== STEP 1: Loading data ===")

features_train = pd.read_csv(
    os.path.join(DATA_DIR, "dengue_features_train.csv"), parse_dates=["week_start_date"]
)
labels_train = pd.read_csv(os.path.join(DATA_DIR, "dengue_labels_train.csv"))
features_test = pd.read_csv(
    os.path.join(DATA_DIR, "dengue_features_test.csv"), parse_dates=["week_start_date"]
)

# Sort by the real calendar date (not year/weekofyear -- a handful of rows
# use an "ISO week 53" label that would sort in the wrong place if we
# trusted the numbers instead of the actual date).
features_train = features_train.sort_values(["city", "week_start_date"]).reset_index(drop=True)
features_test = features_test.sort_values(["city", "week_start_date"]).reset_index(drop=True)

# Attach the case counts (our prediction target) onto the training features.
train_df = features_train.merge(labels_train, on=["city", "year", "weekofyear"], how="left")

print("Training rows:", len(train_df), " Test rows:", len(features_test))
print(train_df.head())

# The climate/vegetation columns we'll use as predictors. Everything else
# (city, year, weekofyear, week_start_date, total_cases) is either an ID
# column or the thing we're trying to predict.
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

# =====================================================================
# STEP 2: Look at the data BEFORE changing anything (exploratory plots)
# =====================================================================
print("\n=== STEP 2: Exploring the raw data ===")

# --- 2a. How do case counts change over time, in each city? ---
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
for ax, city in zip(axes, CITIES):
    city_data = train_df[train_df["city"] == city]
    ax.plot(city_data["week_start_date"], city_data["total_cases"], color="crimson")
    ax.set_title(f"Weekly dengue cases over time -- {city.upper()}")
    ax.set_ylabel("total_cases")
fig.tight_layout()
save_plot(fig, "cases_over_time")

# --- 2b. What does the distribution of weekly case counts look like? ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, city in zip(axes, CITIES):
    city_data = train_df[train_df["city"] == city]
    ax.hist(city_data["total_cases"], bins=30, color="steelblue")
    ax.set_title(f"Distribution of total_cases -- {city.upper()}")
    ax.set_xlabel("total_cases")
fig.tight_layout()
save_plot(fig, "cases_distribution")
# Notice this is very skewed (lots of low-case weeks, a few huge outbreak
# weeks) -- that's why a plain linear regression struggles here, and why
# count-friendly models (Negative Binomial, Poisson-objective boosting) fit
# this kind of data better than they would "normal" continuous data.

# --- 2c. How much data is missing? ---
missing_counts = train_df[CLIMATE_COLUMNS].isna().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(missing_counts.index, missing_counts.values, color="darkorange")
ax.set_title("Missing values per column (training data)")
ax.set_xlabel("number of missing weeks")
fig.tight_layout()
save_plot(fig, "missing_values")

# --- 2d. What do a few of the raw weather variables actually look like? ---
example_cols = ["station_avg_temp_c", "reanalysis_specific_humidity_g_per_kg",
                "precipitation_amt_mm", "ndvi_ne"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flatten(), example_cols):
    ax.hist(train_df[col].dropna(), bins=30, color="seagreen")
    ax.set_title(col)
fig.suptitle("What a few raw input variables look like")
fig.tight_layout()
save_plot(fig, "raw_variable_histograms")

# --- 2e. Which raw variables correlate most with total_cases? ---
correlations = train_df[CLIMATE_COLUMNS + ["total_cases"]].corr()["total_cases"].drop("total_cases")
correlations = correlations.sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["crimson" if v < 0 else "steelblue" for v in correlations.values]
ax.barh(correlations.index, correlations.values, color=colors)
ax.set_title("Correlation of each raw variable with total_cases")
ax.axvline(0, color="black", linewidth=0.8)
fig.tight_layout()
save_plot(fig, "raw_correlation_with_cases")
# These correlations are fairly weak on their own -- that's expected, because
# dengue cases respond to weather with a DELAY (mosquitoes need time to
# breed, people need time to get sick and get diagnosed). STEP 5 builds
# "lag" features specifically to capture that delay, and they correlate
# much more strongly, as you'll see in the next plot.

# =====================================================================
# STEP 3: Stack train + test together so lag features can "see across"
# the train/test boundary (the test weeks immediately follow the training
# weeks in time for both cities)
# =====================================================================
print("\n=== STEP 3: Combining train and test for feature engineering ===")

features_test_with_placeholder = features_test.copy()
features_test_with_placeholder["total_cases"] = np.nan  # unknown, that's what we're predicting

all_data = pd.concat([train_df, features_test_with_placeholder], axis=0, ignore_index=True)
all_data = all_data.sort_values(["city", "week_start_date"]).reset_index(drop=True)

n_train_rows = {city: (train_df["city"] == city).sum() for city in CITIES}

# =====================================================================
# STEP 3: Stack train + test together so lag features can "see across"
# the train/test boundary (the test weeks immediately follow the training
# weeks in time for both cities)
# =====================================================================
print("\n=== STEP 3: Combining train and test for feature engineering ===")

features_test_with_placeholder = features_test.copy()
features_test_with_placeholder["total_cases"] = np.nan  # unknown, that's what we're predicting

all_data = pd.concat([train_df, features_test_with_placeholder], axis=0, ignore_index=True)
all_data = all_data.sort_values(["city", "week_start_date"]).reset_index(drop=True)

n_train_rows = {city: (train_df["city"] == city).sum() for city in CITIES}

# =====================================================================
# STEP 4: Fill in missing values
# =====================================================================
print("\n=== STEP 4: Filling missing values ===")
# Weather doesn't jump around randomly week to week, so a missing value is
# usually very close to the values right before and after it. We use linear
# interpolation (draw a straight line between the known points before/after
# the gap) and, for cases at the very start/end of a city's data, just carry
# the nearest known value forward/backward.

filled_pieces = []
for city in CITIES:
    city_rows = all_data[all_data["city"] == city].copy()
    city_rows[CLIMATE_COLUMNS] = city_rows[CLIMATE_COLUMNS].interpolate(
        method="linear", limit_direction="both"
    )
    city_rows[CLIMATE_COLUMNS] = city_rows[CLIMATE_COLUMNS].ffill().bfill()
    filled_pieces.append(city_rows)
all_data = pd.concat(filled_pieces, axis=0).reset_index(drop=True)

print("Any missing climate values left?", all_data[CLIMATE_COLUMNS].isna().any().any())

# =====================================================================
# STEP 5: Feature engineering -- lag features, rolling averages, season
# =====================================================================
print("\n=== STEP 5: Building lag / rolling / seasonal features ===")
# A "lag" feature is just "what was this variable's value N weeks ago?".
# Dengue outbreaks tend to follow warm, humid weeks by a few weeks (time for
# mosquitoes to breed and bite, and for people to get sick), so a lagged
# version of humidity/temperature is usually a much better predictor than
# the same-week value.

engineered_pieces = []
for city in CITIES:
    city_rows = all_data[all_data["city"] == city].copy()
    city_rows = city_rows.sort_values("week_start_date").reset_index(drop=True)

    # Cyclical encoding of week-of-year: this turns "week 52" and "week 1"
    # into points that are close together (since they ARE close together in
    # the calendar), instead of far apart like the raw numbers 52 and 1.
    city_rows["woy_sin"] = np.sin(2 * np.pi * city_rows["weekofyear"] / 52.0)
    city_rows["woy_cos"] = np.cos(2 * np.pi * city_rows["weekofyear"] / 52.0)

    for col in CLIMATE_COLUMNS:
        for lag in LAG_WEEKS:
            city_rows[f"{col}_lag{lag}"] = city_rows[col].shift(lag)
        for window in ROLLING_WINDOWS:
            # shift(1) first so the rolling average only uses PAST weeks,
            # never the current week -- otherwise we'd be leaking
            # information from the future into our own features.
            city_rows[f"{col}_avg{window}wk"] = (
                city_rows[col].shift(1).rolling(window=window, min_periods=1).mean()
            )

    engineered_pieces.append(city_rows)

all_data = pd.concat(engineered_pieces, axis=0).sort_values(["city", "week_start_date"])
all_data = all_data.reset_index(drop=True)

# The very first few weeks of each city won't have enough history for a lag
# of e.g. 4 or 8 weeks -- fill those remaining gaps the same simple way.
feature_columns = [c for c in all_data.columns
                   if c not in ["city", "year", "weekofyear", "week_start_date", "total_cases"]]
all_data[feature_columns] = all_data.groupby("city")[feature_columns].transform(
    lambda col: col.bfill().ffill()
)

print("Total number of predictor columns:", len(feature_columns))

# --- Plot: do the LAGGED variables correlate better than the raw ones? ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, city in zip(axes, CITIES):
    city_train_rows = all_data[(all_data["city"] == city)].iloc[: n_train_rows[city]]
    cols_to_check = [f"reanalysis_specific_humidity_g_per_kg_lag{lag}" for lag in LAG_WEEKS]
    cols_to_check += ["reanalysis_specific_humidity_g_per_kg"]  # the un-lagged version, for comparison
    corr_values = city_train_rows[cols_to_check + ["total_cases"]].corr()["total_cases"].drop("total_cases")
    ax.bar(corr_values.index, corr_values.values, color="teal")
    ax.set_title(f"Humidity: raw vs. lagged, correlation with cases ({city.upper()})")
    ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
save_plot(fig, "lag_vs_raw_correlation")

# =====================================================================
# STEP 6: Split back into train / test, and (optionally) shrink the
# training set for a fast first run -- see ROWS_PER_CITY at the top
# =====================================================================
print("\n=== STEP 6: Splitting back into train/test ===")

train_final = []
test_final = []
for city in CITIES:
    city_rows = all_data[all_data["city"] == city].sort_values("week_start_date")
    city_train = city_rows.iloc[: n_train_rows[city]]
    city_test = city_rows.iloc[n_train_rows[city]:]

    # A tiny number of weeks in the official data have no case-count label
    # at all (a gap in the original data collection) -- drop those before
    # training since there's nothing to learn from them.
    city_train = city_train[city_train["total_cases"].notna()]

    if ROWS_PER_CITY is not None:
        city_train = city_train.iloc[-ROWS_PER_CITY:]  # keep only the most recent N weeks

    train_final.append(city_train)
    test_final.append(city_test)

train_final = pd.concat(train_final).reset_index(drop=True)
test_final = pd.concat(test_final).reset_index(drop=True)

for city in CITIES:
    n = (train_final["city"] == city).sum()
    print(f"  {city}: training on {n} weeks")

# =====================================================================
# STEP 6: Split back into train / test, and (optionally) shrink the
# training set for a fast first run -- see ROWS_PER_CITY at the top
# =====================================================================
print("\n=== STEP 6: Splitting back into train/test ===")

train_final = []
test_final = []
for city in CITIES:
    city_rows = all_data[all_data["city"] == city].sort_values("week_start_date")
    city_train = city_rows.iloc[: n_train_rows[city]]
    city_test = city_rows.iloc[n_train_rows[city]:]

    # A tiny number of weeks in the official data have no case-count label
    # at all (a gap in the original data collection) -- drop those before
    # training since there's nothing to learn from them.
    city_train = city_train[city_train["total_cases"].notna()]

    if ROWS_PER_CITY is not None:
        city_train = city_train.iloc[-ROWS_PER_CITY:]  # keep only the most recent N weeks

    train_final.append(city_train)
    test_final.append(city_test)

train_final = pd.concat(train_final).reset_index(drop=True)
test_final = pd.concat(test_final).reset_index(drop=True)

for city in CITIES:
    n = (train_final["city"] == city).sum()
    print(f"  {city}: training on {n} weeks")


"""
python dengai_beginner.py

Needs these files in a "data" folder next to this script (download them
from the competitio

n's Data tab, you need a free DrivenData account):
dat a /dengue_features_train.csv
dat a /dengue_labels_train.csv
dat a /dengue_features_test.csv

It will create a "plots" folder full of PNG images, and a "submission.csv"
file with your final predictions.

Python packages needed:
    pip install pandas numpy matplotlib sciki t -learn statsmodels lightgbm
"""





import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
import lightgbm as lgb

# =====================================================================
# SPEED KNOBS -- change these numbers to trade off SPEED vs ACCURACY
# =====================================================================
# While you're still learning / debugging the script, keep these small so
# each run only takes a few seconds. Once you understand what the script
# does and want the best possible accuracy, follow the "for best accuracy"
# notes next to each setting.

# How many of the most recent training weeks to use, per city.
# - Smaller number  = faster runs, but less data for the model to learn from.
# - Set to None     = use ALL available training data (936 weeks for San
#                      Juan, 520 for Iquitos) for best accuracy. This is
#                      still small data by machine-learning standards, so
#                      even "None" trains in well under a minute.
ROWS_PER_CITY = 250  # <-- try None once you're ready for a full run

# How many past weeks we "look back" to build lag features (see STEP 5 for
# what a lag feature is).
# - Fewer lags   = fewer columns = faster.
# - For best accuracy, try LAG_WEEKS = [1, 2, 3, 4, 8]
LAG_WEEKS = [1, 2, 4]

# How many weeks we average over for the rolling-average features.
# - For best accuracy, try ROLLING_WINDOWS = [4, 8]
ROLLING_WINDOWS = [4]

# Random Forest settings (see STEP 7). More trees = slower but usually a
# little more accurate, with diminishing returns past a few hundred.
RF_N_ESTIMATORS = 150  # <-- for best accuracy, try 400-600
RF_MAX_DEPTH = 6  # <-- for best accuracy, try tuning between 4 and 10

# LightGBM settings (see STEP 7). More rounds = slower but usually a little
# more accurate, with diminishing returns past a few hundred.
LGB_N_ESTIMATORS = 200  # <-- for best accuracy, try 400-800
LGB_NUM_LEAVES = 15  # <-- for best accuracy, try 31 (LightGBM's default)
LGB_LEARNING_RATE = 0.05  # <-- smaller learning rate + more rounds usually improves accuracy

# Where things get saved.
DATA_DIR = "data"
PLOTS_DIR = "plots"

RANDOM_STATE = 42
CITIES = ["sj", "iq"]

os.makedirs(PLOTS_DIR, exist_ok=True)
plot_number = [0]  # a little counter so plot filenames sort in the order they were made


def save_plot(fig, name):
    """Save a matplotlib figure into the plots/ folder with a numbered
    filename, e.g. plots/03_missing_values.png, so they're easy to browse
    in the order the script produces them."""
    plot_number[0] += 1
    path = os.path.join(PLOTS_DIR, f"{plot_number[0]:02d}_{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {path}")


# =====================================================================
# STEP 1: Load the raw data
# =====================================================================
print("\n=== STEP 1: Loading data ===")

features_train = pd.read_csv(
    os.path.join(DATA_DIR, "dengue_features_train.csv"), parse_dates=["week_start_date"]
)
labels_train = pd.read_csv(os.path.join(DATA_DIR, "dengue_labels_train.csv"))
features_test = pd.read_csv(
    os.path.join(DATA_DIR, "dengue_features_test.csv"), parse_dates=["week_start_date"]
)

# Sort by the real calendar date (not year/weekofyear -- a handful of rows
# use an "ISO week 53" label that would sort in the wrong place if we
# trusted the numbers instead of the actual date).
features_train = features_train.sort_values(["city", "week_start_date"]).reset_index(drop=True)
features_test = features_test.sort_values(["city", "week_start_date"]).reset_index(drop=True)

# Attach the case counts (our prediction target) onto the training features.
train_df = features_train.merge(labels_train, on=["city", "year", "weekofyear"], how="left")

print("Training rows:", len(train_df), " Test rows:", len(features_test))
print(train_df.head())

# The climate/vegetation columns we'll use as predictors. Everything else
# (city, year, weekofyear, week_start_date, total_cases) is either an ID
# column or the thing we're trying to predict.
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

# =====================================================================
# STEP 2: Look at the data BEFORE changing anything (exploratory plots)
# =====================================================================
print("\n=== STEP 2: Exploring the raw data ===")

# --- 2a. How do case counts change over time, in each city? ---
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
for ax, city in zip(axes, CITIES):
    city_data = train_df[train_df["city"] == city]
    ax.plot(city_data["week_start_date"], city_data["total_cases"], color="crimson")
    ax.set_title(f"Weekly dengue cases over time -- {city.upper()}")
    ax.set_ylabel("total_cases")
fig.tight_layout()
save_plot(fig, "cases_over_time")

# --- 2b. What does the distribution of weekly case counts look like? ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, city in zip(axes, CITIES):
    city_data = train_df[train_df["city"] == city]
    ax.hist(city_data["total_cases"], bins=30, color="steelblue")
    ax.set_title(f"Distribution of total_cases -- {city.upper()}")
    ax.set_xlabel("total_cases")
fig.tight_layout()
save_plot(fig, "cases_distribution")
# Notice this is very skewed (lots of low-case weeks, a few huge outbreak
# weeks) -- that's why a plain linear regression struggles here, and why
# count-friendly models (Negative Binomial, Poisson-objective boosting) fit
# this kind of data better than they would "normal" continuous data.

# --- 2c. How much data is missing? ---
missing_counts = train_df[CLIMATE_COLUMNS].isna().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(missing_counts.index, missing_counts.values, color="darkorange")
ax.set_title("Missing values per column (training data)")
ax.set_xlabel("number of missing weeks")
fig.tight_layout()
save_plot(fig, "missing_values")

# --- 2d. What do a few of the raw weather variables actually look like? ---
example_cols = ["station_avg_temp_c", "reanalysis_specific_humidity_g_per_kg",
                "precipitation_amt_mm", "ndvi_ne"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flatten(), example_cols):
    ax.hist(train_df[col].dropna(), bins=30, color="seagreen")
    ax.set_title(col)
fig.suptitle("What a few raw input variables look like")
fig.tight_layout()
save_plot(fig, "raw_variable_histograms")

# --- 2e. Which raw variables correlate most with total_cases? ---
correlations = train_df[CLIMATE_COLUMNS + ["total_cases"]].corr()["total_cases"].drop("total_cases")
correlations = correlations.sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["crimson" if v < 0 else "steelblue" for v in correlations.values]
ax.barh(correlations.index, correlations.values, color=colors)
ax.set_title("Correlation of each raw variable with total_cases")
ax.axvline(0, color="black", linewidth=0.8)
fig.tight_layout()
save_plot(fig, "raw_correlation_with_cases")
# These correlations are fairly weak on their own -- that's expected, because
# dengue cases respond to weather with a DELAY (mosquitoes need time to
# breed, people need time to get sick and get diagnosed). STEP 5 builds
# "lag" features specifically to capture that delay, and they correlate
# much more strongly, as you'll see in the next plot.

# =====================================================================
# STEP 3: Stack train + test together so lag features can "see across"
# the train/test boundary (the test weeks immediately follow the training
# weeks in time for both cities)
# =====================================================================
print("\n=== STEP 3: Combining train and test for feature engineering ===")

features_test_with_placeholder = features_test.copy()
features_test_with_placeholder["total_cases"] = np.nan  # unknown, that's what we're predicting

all_data = pd.concat([train_df, features_test_with_placeholder], axis=0, ignore_index=True)
all_data = all_data.sort_values(["city", "week_start_date"]).reset_index(drop=True)

n_train_rows = {city: (train_df["city"] == city).sum() for city in CITIES}

# =====================================================================
# STEP 3: Stack train + test together so lag features can "see across"
# the train/test boundary (the test weeks immediately follow the training
# weeks in time for both cities)
# =====================================================================
print("\n=== STEP 3: Combining train and test for feature engineering ===")

features_test_with_placeholder = features_test.copy()
features_test_with_placeholder["total_cases"] = np.nan  # unknown, that's what we're predicting

all_data = pd.concat([train_df, features_test_with_placeholder], axis=0, ignore_index=True)
all_data = all_data.sort_values(["city", "week_start_date"]).reset_index(drop=True)

n_train_rows = {city: (train_df["city"] == city).sum() for city in CITIES}

# =====================================================================
# STEP 4: Fill in missing values
# =====================================================================
print("\n=== STEP 4: Filling missing values ===")
# Weather doesn't jump around randomly week to week, so a missing value is
# usually very close to the values right before and after it. We use linear
# interpolation (draw a straight line between the known points before/after
# the gap) and, for cases at the very start/end of a city's data, just carry
# the nearest known value forward/backward.

filled_pieces = []
for city in CITIES:
    city_rows = all_data[all_data["city"] == city].copy()
    city_rows[CLIMATE_COLUMNS] = city_rows[CLIMATE_COLUMNS].interpolate(
        method="linear", limit_direction="both"
    )
    city_rows[CLIMATE_COLUMNS] = city_rows[CLIMATE_COLUMNS].ffill().bfill()
    filled_pieces.append(city_rows)
all_data = pd.concat(filled_pieces, axis=0).reset_index(drop=True)

print("Any missing climate values left?", all_data[CLIMATE_COLUMNS].isna().any().any())

# =====================================================================
# STEP 5: Feature engineering -- lag features, rolling averages, season
# =====================================================================
print("\n=== STEP 5: Building lag / rolling / seasonal features ===")
# A "lag" feature is just "what was this variable's value N weeks ago?".
# Dengue outbreaks tend to follow warm, humid weeks by a few weeks (time for
# mosquitoes to breed and bite, and for people to get sick), so a lagged
# version of humidity/temperature is usually a much better predictor than
# the same-week value.

engineered_pieces = []
for city in CITIES:
    city_rows = all_data[all_data["city"] == city].copy()
    city_rows = city_rows.sort_values("week_start_date").reset_index(drop=True)

    # Cyclical encoding of week-of-year: this turns "week 52" and "week 1"
    # into points that are close together (since they ARE close together in
    # the calendar), instead of far apart like the raw numbers 52 and 1.
    city_rows["woy_sin"] = np.sin(2 * np.pi * city_rows["weekofyear"] / 52.0)
    city_rows["woy_cos"] = np.cos(2 * np.pi * city_rows["weekofyear"] / 52.0)

    for col in CLIMATE_COLUMNS:
        for lag in LAG_WEEKS:
            city_rows[f"{col}_lag{lag}"] = city_rows[col].shift(lag)
        for window in ROLLING_WINDOWS:
            # shift(1) first so the rolling average only uses PAST weeks,
            # never the current week -- otherwise we'd be leaking
            # information from the future into our own features.
            city_rows[f"{col}_avg{window}wk"] = (
                city_rows[col].shift(1).rolling(window=window, min_periods=1).mean()
            )

    engineered_pieces.append(city_rows)

all_data = pd.concat(engineered_pieces, axis=0).sort_values(["city", "week_start_date"])
all_data = all_data.reset_index(drop=True)

# The very first few weeks of each city won't have enough history for a lag
# of e.g. 4 or 8 weeks -- fill those remaining gaps the same simple way.
feature_columns = [c for c in all_data.columns
                   if c not in ["city", "year", "weekofyear", "week_start_date", "total_cases"]]
all_data[feature_columns] = all_data.groupby("city")[feature_columns].transform(
    lambda col: col.bfill().ffill()
)

print("Total number of predictor columns:", len(feature_columns))

# --- Plot: do the LAGGED variables correlate better than the raw ones? ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, city in zip(axes, CITIES):
    city_train_rows = all_data[(all_data["city"] == city)].iloc[: n_train_rows[city]]
    cols_to_check = [f"reanalysis_specific_humidity_g_per_kg_lag{lag}" for lag in LAG_WEEKS]
    cols_to_check += ["reanalysis_specific_humidity_g_per_kg"]  # the un-lagged version, for comparison
    corr_values = city_train_rows[cols_to_check + ["total_cases"]].corr()["total_cases"].drop("total_cases")
    ax.bar(corr_values.index, corr_values.values, color="teal")
    ax.set_title(f"Humidity: raw vs. lagged, correlation with cases ({city.upper()})")
    ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
save_plot(fig, "lag_vs_raw_correlation")

# =====================================================================
# STEP 6: Split back into train / test, and (optionally) shrink the
# training set for a fast first run -- see ROWS_PER_CITY at the top
# =====================================================================
print("\n=== STEP 6: Splitting back into train/test ===")

train_final = []
test_final = []
for city in CITIES:
    city_rows = all_data[all_data["city"] == city].sort_values("week_start_date")
    city_train = city_rows.iloc[: n_train_rows[city]]
    city_test = city_rows.iloc[n_train_rows[city]:]

    # A tiny number of weeks in the official data have no case-count label
    # at all (a gap in the original data collection) -- drop those before
    # training since there's nothing to learn from them.
    city_train = city_train[city_train["total_cases"].notna()]

    if ROWS_PER_CITY is not None:
        city_train = city_train.iloc[-ROWS_PER_CITY:]  # keep only the most recent N weeks

    train_final.append(city_train)
    test_final.append(city_test)

train_final = pd.concat(train_final).reset_index(drop=True)
test_final = pd.concat(test_final).reset_index(drop=True)

for city in CITIES:
    n = (train_final["city"] == city).sum()
    print(f"  {city}: training on {n} weeks")

# =====================================================================
# STEP 6: Split back into train / test, and (optionally) shrink the
# training set for a fast first run -- see ROWS_PER_CITY at the top
# =====================================================================
print("\n=== STEP 6: Splitting back into train/test ===")

train_final = []
test_final = []
for city in CITIES:
    city_rows = all_data[all_data["city"] == city].sort_values("week_start_date")
    city_train = city_rows.iloc[: n_train_rows[city]]
    city_test = city_rows.iloc[n_train_rows[city]:]

    # A tiny number of weeks in the official data have no case-count label
    # at all (a gap in the original data collection) -- drop those before
    # training since there's nothing to learn from them.
    city_train = city_train[city_train["total_cases"].notna()]

    if ROWS_PER_CITY is not None:
        city_train = city_train.iloc[-ROWS_PER_CITY:]  # keep only the most recent N weeks

    train_final.append(city_train)
    test_final.append(city_test)

train_final = pd.concat(train_final).reset_index(drop=True)
test_final = pd.concat(test_final).reset_index(drop=True)

for city in CITIES:
    n = (train_final["city"] == city).sum()
    print(f"  {city}: training on {n} weeks")

