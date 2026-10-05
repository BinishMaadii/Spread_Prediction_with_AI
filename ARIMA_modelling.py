# dengai_arima.py
# ARIMA model for the DengAI project.
# It uses the SAME walk-forward folds and the SAME metrics as the ML pipeline,
# so the numbers can be compared directly with GLM, Random Forest and LightGBM.
#
# Run order:
#   1) your ML pipeline (creates cv_results_by_fold.csv)
#   2) this file (creates arima_results_by_fold.csv)
#   3) dengai_sarima.py (creates sarima_results_by_fold.csv)

import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

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

# ARIMA orders to try. d is NOT in this grid (see Step 2 for why).
P_GRID = [0, 1, 2, 3]
Q_GRID = [0, 1, 2]

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
# STEP 2: Walk-forward evaluation of ARIMA
# For each validation year:
#   a) take only the weeks BEFORE that year as history
#   b) take log(1 + cases) so big outbreaks do not dominate the fit
#   c) choose d with an ADF stationarity test
#   d) choose p and q with the lowest AIC
#   e) forecast the whole next year at once (52 weeks ahead)
#
# Why d is chosen by a test and not by AIC:
# AIC values of models with different d are not comparable,
# because the models are fitted to different (differenced) series.
# =====================================================================
print("\n=== STEP 2: ARIMA walk-forward evaluation ===")
results = []
last_fold = {}

for city in CITIES:
    city_rows = train_df[train_df["city"] == city].reset_index(drop=True)
    cases = city_rows["total_cases"].values.astype(float)
    dates = city_rows["week_start_date"]
    n = len(cases)
    print(f"  {city.upper()}: {n} weeks")

    for fold in range(N_FOLDS):
        val_start = n - (N_FOLDS - fold) * HORIZON
        history = cases[:val_start]
        actual = cases[val_start:val_start + HORIZON]
        history_log = np.log1p(history)

        # c) ADF test: p-value below 0.05 means the series is stationary, so d = 0
        adf_p_value = adfuller(history_log)[1]
        d = 0 if adf_p_value < 0.05 else 1

        # d) grid search over p and q
        best_aic = np.inf
        best_fit = None
        best_order = None
        for p, q in product(P_GRID, Q_GRID):
            try:
                fit = ARIMA(history_log, order=(p, d, q)).fit()
            except Exception:
                continue
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
                best_order = (p, d, q)

        # e) forecast the validation year and go back from the log scale
        forecast_log = best_fit.forecast(steps=HORIZON)
        predicted = np.clip(np.expm1(forecast_log), 0, None)

        mae, rmse, mase, peak_mae = score(actual, predicted, history)
        results.append({
            "setup": "arima", "fold": fold + 1, "model": "arima", "city": city,
            "mae": mae, "rmse": rmse, "mase": mase, "peak_mae": peak_mae,
            "order": str(best_order), "adf_p_value": round(adf_p_value, 4),
        })
        print(f"    fold {fold + 1}: order {best_order}, ADF p = {adf_p_value:.3f}, "
              f"MAE = {mae:.2f}, MASE = {mase:.3f}")
        last_fold[city] = (dates.iloc[val_start:val_start + HORIZON], actual, predicted)

results_df = pd.DataFrame(results)
results_df.to_csv("arima_results_by_fold.csv", index=False)
print("\n  saved -> arima_results_by_fold.csv")

# =====================================================================
# STEP 3: Plot the last validation year of each city
# =====================================================================
print("\n=== STEP 3: Plots ===")
fig, axes = plt.subplots(2, 1, figsize=(10, 7))
for ax, city in zip(axes, CITIES):
    week_dates, actual, predicted = last_fold[city]
    ax.plot(week_dates, actual, color="black", linewidth=2, label="actual")
    ax.plot(week_dates, predicted, color="steelblue", linestyle="--", label="ARIMA")
    ax.set_title(f"{city.upper()}: ARIMA forecast, last validation year")
    ax.set_ylabel("total_cases")
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "arima_last_year.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("  saved plot -> plots/arima_last_year.png")

# =====================================================================
# STEP 4: Compare with the ML models (and with SARIMA if it was already run)
# =====================================================================
print("\n=== STEP 4: Comparison ===")
columns = ["city", "fold", "model", "mae", "rmse", "mase", "peak_mae"]
frames = []

if os.path.exists("cv_results_by_fold.csv"):
    ml_results = pd.read_csv("cv_results_by_fold.csv")
    ml_results = ml_results[ml_results["setup"] == "after"]
    frames.append(ml_results[columns])
else:
    print("  cv_results_by_fold.csv not found, run the ML pipeline first to compare.")

frames.append(results_df[columns])
if os.path.exists("sarima_results_by_fold.csv"):
    frames.append(pd.read_csv("sarima_results_by_fold.csv")[columns])

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
