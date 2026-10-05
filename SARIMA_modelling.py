# dengai_sarima.py
# SARIMA and SARIMAX models for the DengAI project.
# Same walk-forward folds and same metrics as the ML pipeline and dengai_arima.py.
#
# Two seasonal models are compared:
#   sarima_52        true SARIMA, seasonal cycle of 52 weeks, uses only past cases
#   sarimax_fourier  SARIMAX: ARIMA errors + Fourier seasonality + 2 climate inputs
#
# Run order:
#   1) your ML pipeline (creates cv_results_by_fold.csv)
#   2) dengai_arima.py (creates arima_results_by_fold.csv)
#   3) this file (creates sarima_results_by_fold.csv and the final comparison)

import os
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# =====================================================================
# SETTINGS
# =====================================================================
# Local files: put the CSV files in a folder called "data" next to this script.
# All output files (csv, plots) are also written next to this script.
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
os.chdir(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")  # change this line if your CSVs are elsewhere

if not os.path.exists(os.path.join(DATA_DIR, "dengue_features_train.csv")):
    raise FileNotFoundError(
        "Could not find dengue_features_train.csv and dengue_labels_train.csv in "
        f"{DATA_DIR}. Create that folder and copy the DrivenData CSV files into it.")
PLOTS_DIR = "plots"
CITIES = ["sj", "iq"]

# Same folds as the ML pipeline: 4 validation years of 52 weeks each
N_FOLDS = 4
HORIZON = 52

# True SARIMA: (p,d,q) x (P,D,Q,m). One seasonal difference at lag 52, plus
# one AR and one MA term for the weekly pattern and one seasonal MA term.
# The order is fixed because a grid search at m = 52 is very slow.
SARIMA_ORDER = (1, 0, 1)
SARIMA_SEASONAL_ORDER = (0, 1, 1, 52)

# SARIMAX: small AIC grid. d = 0 because the Fourier terms already remove
# the yearly pattern, so the errors are close to stationary.
P_GRID = [1, 2]
Q_GRID = [0, 1]

# The two climate inputs are the same ones your GLM uses, lagged by 2 weeks
CLIMATE_COLUMNS = ["reanalysis_specific_humidity_g_per_kg", "station_avg_temp_c"]
LAG = 2

os.makedirs(PLOTS_DIR, exist_ok=True)


def score(actual, predicted, history):
    # Same four metrics as forecast_metrics() in the ML pipeline
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    naive_mae = np.mean(np.abs(history[52:] - history[:-52]))
    mase = mae / naive_mae
    peak_weeks = actual >= np.percentile(history, 90)
    if peak_weeks.sum() > 0:
        peak_mae = mean_absolute_error(actual[peak_weeks], predicted[peak_weeks])
    else:
        peak_mae = np.nan
    return mae, rmse, mase, peak_mae


# =====================================================================
# STEP 1: Load the training data
# =====================================================================
print("\n=== STEP 1: Loading data ===")
features = pd.read_csv(os.path.join(DATA_DIR, "dengue_features_train.csv"),
                       parse_dates=["week_start_date"])
labels = pd.read_csv(os.path.join(DATA_DIR, "dengue_labels_train.csv"))
train_df = features.merge(labels, on=["city", "year", "weekofyear"], how="left")
train_df = train_df.sort_values(["city", "week_start_date"]).reset_index(drop=True)
print("Training rows:", len(train_df))

# =====================================================================
# STEP 2: Build the inputs for SARIMAX (per city)
# - fill gaps with the last known value (past only, no leakage)
# - Fourier terms: 2 sine/cosine pairs for the yearly cycle
# - climate columns shifted by 2 weeks
# The validation weeks use their own climate values, exactly like the ML models do.
# =====================================================================
print("\n=== STEP 2: Building SARIMAX inputs ===")
exog_by_city = {}
for city in CITIES:
    city_rows = train_df[train_df["city"] == city].reset_index(drop=True)
    climate = city_rows[CLIMATE_COLUMNS].ffill().bfill()
    climate = climate.shift(LAG).bfill()

    angle = 2 * np.pi * city_rows["weekofyear"] / 52.0
    exog = pd.DataFrame({
        "sin1": np.sin(angle), "cos1": np.cos(angle),
        "sin2": np.sin(2 * angle), "cos2": np.cos(2 * angle),
    })
    exog = pd.concat([exog, climate.add_suffix(f"_lag{LAG}")], axis=1)
    exog_by_city[city] = exog.values.astype(float)
    print(f"  {city.upper()}: {exog.shape[1]} input columns ->", list(exog.columns))

# =====================================================================
# STEP 3: Walk-forward evaluation
# Both models forecast the whole next year at once (52 weeks ahead),
# the same way the ML models do.
# =====================================================================
print("\n=== STEP 3: SARIMA and SARIMAX walk-forward evaluation ===")
results = []
last_fold = {}

for city in CITIES:
    city_rows = train_df[train_df["city"] == city].reset_index(drop=True)
    cases = city_rows["total_cases"].values.astype(float)
    dates = city_rows["week_start_date"]
    exog_all = exog_by_city[city]
    n = len(cases)
    print(f"  {city.upper()}: {n} weeks")

    for fold in range(N_FOLDS):
        val_start = n - (N_FOLDS - fold) * HORIZON
        history = cases[:val_start]
        actual = cases[val_start:val_start + HORIZON]
        history_log = np.log1p(history)
        predictions = {}

        # ---- Model 1: true SARIMA, past cases only ----
        start = time.time()
        sarima_fit = SARIMAX(history_log, order=SARIMA_ORDER,
                             seasonal_order=SARIMA_SEASONAL_ORDER).fit(disp=False, maxiter=100)
        predictions["sarima_52"] = np.clip(np.expm1(sarima_fit.forecast(steps=HORIZON)), 0, None)
        sarima_seconds = time.time() - start

        # ---- Model 2: SARIMAX with Fourier terms and climate inputs ----
        # Scale the inputs with the training mean and std only
        x_fit = exog_all[:val_start]
        x_val = exog_all[val_start:val_start + HORIZON]
        x_mean = x_fit.mean(axis=0)
        x_std = x_fit.std(axis=0)
        x_std[x_std == 0] = 1
        x_fit = (x_fit - x_mean) / x_std
        x_val = (x_val - x_mean) / x_std

        best_aic = np.inf
        best_fit = None
        best_order = None
        for p, q in product(P_GRID, Q_GRID):
            try:
                fit = SARIMAX(history_log, exog=x_fit, order=(p, 0, q),
                              trend="c").fit(disp=False, maxiter=100)
            except Exception:
                continue
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
                best_order = (p, 0, q)
        forecast_log = best_fit.forecast(steps=HORIZON, exog=x_val)
        predictions["sarimax_fourier"] = np.clip(np.expm1(forecast_log), 0, None)

        for model_name, predicted in predictions.items():
            mae, rmse, mase, peak_mae = score(actual, predicted, history)
            results.append({
                "setup": "sarima", "fold": fold + 1, "model": model_name, "city": city,
                "mae": mae, "rmse": rmse, "mase": mase, "peak_mae": peak_mae,
            })
        print(f"    fold {fold + 1}: sarima_52 took {sarima_seconds:.0f}s, "
              f"sarimax order {best_order}")
        for model_name, predicted in predictions.items():
            print(f"       {model_name:16s} MAE = {mean_absolute_error(actual, predicted):.2f}")
        last_fold[city] = (dates.iloc[val_start:val_start + HORIZON], actual, predictions)

results_df = pd.DataFrame(results)
results_df.to_csv("sarima_results_by_fold.csv", index=False)
print("\n  saved -> sarima_results_by_fold.csv")

# =====================================================================
# STEP 4: Plot the last validation year of each city
# =====================================================================
print("\n=== STEP 4: Plots ===")
fig, axes = plt.subplots(2, 1, figsize=(10, 7))
for ax, city in zip(axes, CITIES):
    week_dates, actual, predictions = last_fold[city]
    ax.plot(week_dates, actual, color="black", linewidth=2, label="actual")
    ax.plot(week_dates, predictions["sarima_52"], linestyle="--", label="SARIMA (m=52)")
    ax.plot(week_dates, predictions["sarimax_fourier"], linestyle="--", label="SARIMAX (Fourier + climate)")
    ax.set_title(f"{city.upper()}: seasonal models, last validation year")
    ax.set_ylabel("total_cases")
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "sarima_last_year.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("  saved plot -> plots/sarima_last_year.png")

# =====================================================================
# STEP 5: Final comparison: ML models vs ARIMA vs SARIMA vs SARIMAX
# =====================================================================
print("\n=== STEP 5: Comparison ===")
columns = ["city", "fold", "model", "mae", "rmse", "mase", "peak_mae"]
frames = []

if os.path.exists("cv_results_by_fold.csv"):
    ml_results = pd.read_csv("cv_results_by_fold.csv")
    ml_results = ml_results[ml_results["setup"] == "after"]
    frames.append(ml_results[columns])
else:
    print("  cv_results_by_fold.csv not found, run the ML pipeline first to compare.")

if os.path.exists("arima_results_by_fold.csv"):
    frames.append(pd.read_csv("arima_results_by_fold.csv")[columns])
else:
    print("  arima_results_by_fold.csv not found, run dengai_arima.py to include ARIMA.")

frames.append(results_df[columns])

compare = pd.concat(frames, ignore_index=True)
summary = compare.groupby(["city", "model"])[["mae", "rmse", "mase", "peak_mae"]].mean()

for city in CITIES:
    print(f"\n{city.upper()}: average over {N_FOLDS} validation years (lower is better)")
    print(summary.loc[city].sort_values("mae").round(2).to_string())

summary.round(3).to_csv("model_comparison.csv")
print("\n  saved -> model_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, city in zip(axes, CITIES):
    city_summary = summary.loc[city].sort_values("mae")
    ax.bar(city_summary.index, city_summary["mae"], color="steelblue")
    ax.set_title(f"{city.upper()}: average MAE over validation years")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=40)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("  saved plot -> plots/model_comparison.png")
