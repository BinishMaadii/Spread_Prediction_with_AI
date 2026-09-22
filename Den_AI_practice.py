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
 




# =====================================================================
# STEP 2: A few exploratory plots of the raw data
# =====================================================================
print("\n=== STEP 2: Exploring the raw data ===")
 
fig, axes = plt.subplots(2, 1, figsize=(11, 6))
for ax, city in zip(axes, CITIES):
    city_data = train_df[train_df["city"] == city]
    ax.plot(city_data["week_start_date"], city_data["total_cases"], color="crimson")
    ax.set_title(f"Weekly dengue cases over time, {city.upper()}")
    ax.set_ylabel("total_cases")
fig.tight_layout()
save_plot(fig, "cases_over_time")
 
missing_counts = train_df[CLIMATE_COLUMNS].isna().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(missing_counts.index, missing_counts.values, color="darkorange")
ax.set_title("Missing values per column (training data)")
fig.tight_layout()
save_plot(fig, "missing_values")
 
correlations = train_df[CLIMATE_COLUMNS + ["total_cases"]].corr()["total_cases"].drop("total_cases")
correlations = correlations.sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["crimson" if v < 0 else "steelblue" for v in correlations.values]
ax.barh(correlations.index, correlations.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Correlation of each raw variable with total_cases")
fig.tight_layout()
save_plot(fig, "raw_correlation_with_cases")
 
 
# =====================================================================
# Data preparation functions (filling + feature engineering)
# =====================================================================
def fill_missing(all_data, method):
    pieces = []
    for city in CITIES:
        city_rows = all_data[all_data["city"] == city].copy()
        if method == "interpolate":
            # Original method. It looks at the week AFTER a gap too,
            # which you would not have yet in a real forecast.
            city_rows[CLIMATE_COLUMNS] = city_rows[CLIMATE_COLUMNS].interpolate(
                method="linear", limit_direction="both")
        # "past_only": copy the last known value forward.
        # The bfill only affects the first few rows of each city.
        city_rows[CLIMATE_COLUMNS] = city_rows[CLIMATE_COLUMNS].ffill().bfill()
        pieces.append(city_rows)
    return pd.concat(pieces).reset_index(drop=True)
 
 
def add_features(all_data, lags, windows):
    pieces = []
    for city in CITIES:
        city_rows = all_data[all_data["city"] == city].copy()
        city_rows = city_rows.sort_values("week_start_date").reset_index(drop=True)
 
        city_rows["woy_sin"] = np.sin(2 * np.pi * city_rows["weekofyear"] / 52.0)
        city_rows["woy_cos"] = np.cos(2 * np.pi * city_rows["weekofyear"] / 52.0)
 
        for col in CLIMATE_COLUMNS:
            for lag in lags:
                city_rows[f"{col}_lag{lag}"] = city_rows[col].shift(lag)
            for window in windows:
                city_rows[f"{col}_avg{window}wk"] = (
                    city_rows[col].shift(1).rolling(window=window, min_periods=1).mean()
                )
 
        # The first weeks of each city have no history for the lags yet
        feature_cols = [c for c in city_rows.columns if c not in ID_COLUMNS]
        city_rows[feature_cols] = city_rows[feature_cols].bfill().ffill()
        pieces.append(city_rows)
    return pd.concat(pieces).reset_index(drop=True)
 
 
def prepare_data(setup):
    """Returns two dicts: city -> training rows, city -> test rows."""
    test_copy = features_test.copy()
    test_copy["total_cases"] = np.nan
    all_data = pd.concat([train_df, test_copy], ignore_index=True)
    all_data = all_data.sort_values(["city", "week_start_date"]).reset_index(drop=True)
 
    all_data = fill_missing(all_data, setup["fill_method"])
    all_data = add_features(all_data, setup["lags"], setup["windows"])
 
    train_by_city = {}
    test_by_city = {}
    for city in CITIES:
        city_rows = all_data[all_data["city"] == city].reset_index(drop=True)
        n_train = (train_df["city"] == city).sum()
        city_train = city_rows.iloc[:n_train]
        city_train = city_train[city_train["total_cases"].notna()].reset_index(drop=True)
        train_by_city[city] = city_train
        test_by_city[city] = city_rows.iloc[n_train:].reset_index(drop=True)
    return train_by_city, test_by_city
 
 
# =====================================================================
# Model functions
# =====================================================================
def get_X(rows):
    return rows.drop(columns=ID_COLUMNS)
 
 
def fit_negbin(fit_rows, alpha):
    formula = "total_cases ~ " + " + ".join(GLM_COLUMNS)
    data = fit_rows[GLM_COLUMNS + ["total_cases"]]
    family = sm.families.NegativeBinomial(alpha=alpha)
    return smf.glm(formula=formula, data=data, family=family).fit()
 
 
def fit_random_forest(fit_rows, setup):
    model = RandomForestRegressor(
        n_estimators=setup["rf_trees"],
        max_depth=setup["rf_depth"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(get_X(fit_rows), fit_rows["total_cases"])
    return model
 
 
def fit_lightgbm(fit_rows, setup, quantile=None):
    # quantile=None -> normal Poisson model that predicts the expected count
    # quantile=0.1  -> model that predicts the 10th percentile, and so on
    if quantile is None:
        model = lgb.LGBMRegressor(
            objective="poisson",
            n_estimators=setup["lgb_rounds"],
            num_leaves=setup["lgb_leaves"],
            learning_rate=setup["lgb_learning_rate"],
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    else:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=setup["lgb_rounds"],
            num_leaves=setup["lgb_leaves"],
            learning_rate=setup["lgb_learning_rate"],
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    model.fit(get_X(fit_rows), fit_rows["total_cases"])
    return model
 
 
def predict(model, rows, is_glm=False):
    if is_glm:
        preds = model.predict(rows[GLM_COLUMNS])
    else:
        preds = model.predict(get_X(rows))
    return np.clip(np.asarray(preds, dtype=float), 0, None)
 
 
def predict_seasonal_median(fit_rows, val_rows):
    # Baseline: "this week will look like the median of the same week in past years"
    median_by_week = fit_rows.groupby("weekofyear")["total_cases"].median()
    preds = val_rows["weekofyear"].map(median_by_week)
    preds = preds.fillna(fit_rows["total_cases"].median())  # e.g. week 53
    return preds.values.astype(float)
 
 
def predict_last_value(fit_rows, val_rows):
    # Baseline: "every future week will have the same count as the last known week"
    return np.full(len(val_rows), float(fit_rows["total_cases"].iloc[-1]))
 
 
def tune_negbin_alpha(fit_rows):
    """Try every alpha on the LAST year of the training rows and keep the best.
    Only training rows are used, so the validation year stays unseen."""
    inner_fit = fit_rows.iloc[:-HORIZON]
    inner_val = fit_rows.iloc[-HORIZON:]
    best_alpha = 1.0
    best_mae = float("inf")
    for alpha in ALPHA_GRID:
        try:
            model = fit_negbin(inner_fit, alpha)
            preds = predict(model, inner_val, is_glm=True)
        except Exception:
            continue
        if not np.all(np.isfinite(preds)):
            continue
        mae = mean_absolute_error(inner_val["total_cases"], preds)
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    return best_alpha
 
 
# =====================================================================
# Metrics
# =====================================================================
def forecast_metrics(actual, predicted, history):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    history = np.asarray(history, dtype=float)
 
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
 
    # MASE: our MAE divided by the MAE of a seasonal-naive forecast
    # ("same week last year") on the training history.
    # MASE below 1 means we beat the naive forecast.
    season = 52 if len(history) > 52 else 1
    naive_mae = np.mean(np.abs(history[season:] - history[:-season]))
    mase = mae / naive_mae
 
    # Peak-week MAE: error only on outbreak weeks
    # (weeks above the 90th percentile of the training history)
    peak_weeks = actual >= np.percentile(history, 90)
    if peak_weeks.sum() > 0:
        peak_mae = mean_absolute_error(actual[peak_weeks], predicted[peak_weeks])
    else:
        peak_mae = np.nan
 
    return {"mae": mae, "rmse": rmse, "mase": mase, "peak_mae": peak_mae}
 
 
def interval_coverage(fit_rows, val_rows, setup):
    """80% prediction interval from two LightGBM quantile models.
    Coverage = share of real weeks that fall inside the interval (target: 0.80)."""
    low = predict(fit_lightgbm(fit_rows, setup, quantile=0.1), val_rows)
    high = predict(fit_lightgbm(fit_rows, setup, quantile=0.9), val_rows)
    actual = val_rows["total_cases"].values
    inside = (actual >= low) & (actual <= high)
    return inside.mean(), (high - low).mean()
 
 
# =====================================================================
# Walk-forward evaluation
# =====================================================================
def get_all_predictions(fit_rows, val_rows, setup):
    alpha = setup["negbin_alpha"]
    if alpha == "tune":
        alpha = tune_negbin_alpha(fit_rows)
 
    preds = {}
    preds["seasonal_median"] = predict_seasonal_median(fit_rows, val_rows)
    preds["last_value"] = predict_last_value(fit_rows, val_rows)
    preds["negbin_glm"] = predict(fit_negbin(fit_rows, alpha), val_rows, is_glm=True)
    preds["random_forest"] = predict(fit_random_forest(fit_rows, setup), val_rows)
    preds["lightgbm"] = predict(fit_lightgbm(fit_rows, setup), val_rows)
    if setup["use_ensemble"]:
        preds["ensemble"] = (preds["negbin_glm"] + preds["lightgbm"]) / 2
    return preds, alpha
 
 
def walk_forward(city_train, setup):
    results = []
    last_fold = None
    n = len(city_train)
 
    for fold in range(N_FOLDS):
        val_start = n - (N_FOLDS - fold) * HORIZON
        fit_rows = city_train.iloc[:val_start]
        if setup["rows_per_city"] is not None:
            fit_rows = fit_rows.iloc[-setup["rows_per_city"]:]
        val_rows = city_train.iloc[val_start:val_start + HORIZON]
 
        preds, alpha = get_all_predictions(fit_rows, val_rows, setup)
        coverage, width = interval_coverage(fit_rows, val_rows, setup)
 
        for model_name in preds:
            full_history = city_train.iloc[:val_start]["total_cases"]
            metrics = forecast_metrics(val_rows["total_cases"], preds[model_name],
                                       fit_rows["total_cases"])
            metrics["setup"] = setup["name"]
            metrics["fold"] = fold + 1
            metrics["model"] = model_name
            metrics["val_start"] = val_rows["week_start_date"].iloc[0].date()
            if model_name == "lightgbm":
                metrics["coverage_80"] = coverage
                metrics["interval_width"] = width
            else:
                metrics["coverage_80"] = np.nan
                metrics["interval_width"] = np.nan
            results.append(metrics)
 
        print(f"    fold {fold + 1}: validation year starts {val_rows['week_start_date'].iloc[0].date()}, "
              f"trained on {len(fit_rows)} weeks, negbin alpha = {alpha}")
        last_fold = (val_rows, preds)
 
    return pd.DataFrame(results), last_fold







# =====================================================================
# STEP 3: Evaluate BEFORE and AFTER on the same validation years to see the impact of improvments
# =====================================================================
all_results = []
last_folds = {}
data_by_setup = {}
 
for setup in [BEFORE, AFTER]:
    print(f"\n=== STEP 3: Walk-forward evaluation, setup = {setup['name'].upper()} ===")
    train_by_city, test_by_city = prepare_data(setup)
    data_by_setup[setup["name"]] = (train_by_city, test_by_city)
    for city in CITIES:
        print(f"  {city.upper()}:")
        city_results, last_fold = walk_forward(train_by_city[city], setup)
        city_results["city"] = city
        all_results.append(city_results)
        last_folds[(setup["name"], city)] = last_fold
 
results_df = pd.concat(all_results, ignore_index=True)
results_df.to_csv("cv_results_by_fold.csv", index=False)
print("\n  saved every fold's scores -> cv_results_by_fold.csv")





# =====================================================================
# STEP 4: Compare BEFORE vs AFTER
# =====================================================================
print("\n=== STEP 4: Before vs. after ===")
 
best_model = {}  # (city, setup_name) -> name of the best real model
for city in CITIES:
    for setup_name in ["before", "after"]:
        rows = results_df[(results_df["city"] == city) &
                          (results_df["setup"] == setup_name) &
                          (results_df["model"].isin(REAL_MODELS))]
        mean_mae = rows.groupby("model")["mae"].mean()
        best_model[(city, setup_name)] = mean_mae.idxmin()
 
experiment_rows = []
for city in CITIES:
    city_results = results_df[results_df["city"] == city]
 
    print(f"\n{city.upper()}: average MAE over {N_FOLDS} validation years (lower is better)")
    table = city_results.pivot_table(index="model", columns="setup", values="mae", aggfunc="mean")
    table = table[["before", "after"]]
    print(table.round(2).to_string())
 
    for setup_name in ["before", "after"]:
        winner = best_model[(city, setup_name)]
        rows = city_results[(city_results["setup"] == setup_name) & (city_results["model"] == winner)]
        baseline_rows = city_results[(city_results["setup"] == setup_name) &
                                     (city_results["model"] == "seasonal_median")]
        lgb_rows = city_results[(city_results["setup"] == setup_name) &
                                (city_results["model"] == "lightgbm")]
        setup = BEFORE if setup_name == "before" else AFTER
        experiment_rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "city": city,
            "setup": setup_name,
            "rows_per_city": setup["rows_per_city"],
            "fill_method": setup["fill_method"],
            "lags": str(setup["lags"]),
            "windows": str(setup["windows"]),
            "best_model": winner,
            "mae_mean": round(rows["mae"].mean(), 2),
            "mae_std": round(rows["mae"].std(), 2),
            "rmse": round(rows["rmse"].mean(), 2),
            "mase": round(rows["mase"].mean(), 3),
            "peak_mae": round(rows["peak_mae"].mean(), 2),
            "baseline_mae": round(baseline_rows["mae"].mean(), 2),
            "lgb_coverage_80": round(lgb_rows["coverage_80"].mean(), 2),
        })
 
    before = experiment_rows[-2]
    after = experiment_rows[-1]
    change = 100 * (after["mae_mean"] - before["mae_mean"]) / before["mae_mean"]
    print(f"\n  BEFORE best model: {before['best_model']:14s} MAE = {before['mae_mean']:.2f} "
          f"+/- {before['mae_std']:.2f}, MASE = {before['mase']:.3f}, peak MAE = {before['peak_mae']:.2f}")
    print(f"  AFTER  best model: {after['best_model']:14s} MAE = {after['mae_mean']:.2f} "
          f"+/- {after['mae_std']:.2f}, MASE = {after['mase']:.3f}, peak MAE = {after['peak_mae']:.2f}")
    print(f"  Change in MAE: {change:+.1f}%   (seasonal-median baseline MAE = {after['baseline_mae']:.2f})")
    print(f"  LightGBM 80% interval coverage: before {before['lgb_coverage_80']:.0%}, "
          f"after {after['lgb_coverage_80']:.0%} (80% means well calibrated)")
 
experiments = pd.DataFrame(experiment_rows)
write_header = not os.path.exists("experiments.csv")
experiments.to_csv("experiments.csv", mode="a", header=write_header, index=False)
print("\n  appended summary -> experiments.csv")
 
# --- Plot: MAE per model, before vs after ---
model_order = ["seasonal_median", "last_value", "negbin_glm", "random_forest", "lightgbm", "ensemble"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, city in zip(axes, CITIES):
    city_results = results_df[results_df["city"] == city]
    x = np.arange(len(model_order))
    for i, setup_name in enumerate(["before", "after"]):
        mean_mae = city_results[city_results["setup"] == setup_name].groupby("model")["mae"].mean()
        values = [mean_mae.get(m, np.nan) for m in model_order]
        ax.bar(x + i * 0.4, values, width=0.4, label=setup_name)
    ax.set_xticks(x + 0.2)
    ax.set_xticklabels(model_order, rotation=30)
    ax.set_ylabel("average MAE over validation years")
    ax.set_title(f"{city.upper()}: before vs. after")
    ax.legend()
fig.tight_layout()
save_plot(fig, "before_vs_after_mae")
 
# --- Plot: MAE of the best model in each validation year ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, city in zip(axes, CITIES):
    for setup_name in ["before", "after"]:
        winner = best_model[(city, setup_name)]
        rows = results_df[(results_df["city"] == city) & (results_df["setup"] == setup_name) &
                          (results_df["model"] == winner)]
        ax.plot(rows["fold"], rows["mae"], marker="o", label=f"{setup_name} ({winner})")
    ax.set_xlabel("validation year (fold)")
    ax.set_ylabel("MAE")
    ax.set_title(f"{city.upper()}: MAE in each validation year")
    ax.legend()
fig.tight_layout()
save_plot(fig, "mae_per_fold")
 
# --- Plot: last validation year, actual vs. before vs. after ---
for city in CITIES:
    val_rows, before_preds = last_folds[("before", city)]
    _, after_preds = last_folds[("after", city)]
    before_winner = best_model[(city, "before")]
    after_winner = best_model[(city, "after")]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(val_rows["week_start_date"], val_rows["total_cases"], color="black", linewidth=2, label="actual")
    ax.plot(val_rows["week_start_date"], before_preds[before_winner], linestyle="--",
            label=f"before ({before_winner})")
    ax.plot(val_rows["week_start_date"], after_preds[after_winner], linestyle="--",
            label=f"after ({after_winner})")
    ax.set_title(f"{city.upper()}: last validation year")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, f"{city}_last_year_before_vs_after")









