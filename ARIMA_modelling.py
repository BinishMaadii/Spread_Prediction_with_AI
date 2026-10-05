
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
DATA_DIR = "/kaggle/input/datasets/binishbatool/dengue-time-series"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "data"
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
