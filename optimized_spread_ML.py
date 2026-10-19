

import glob
import json
import os
import time
 
import matplotlib
matplotlib.use("Agg")   # save figures only, do not open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
 
# ---- config ----------------------------------------------------------------
DATA_DIR = os.environ.get("WM811K_DIR", "/kaggle/input/datasets/qingyi/wm811k-wafer-map")
OUT_DIR = os.environ.get("OUT_DIR", "outputs_v2")
EPOCHS = int(os.environ.get("EPOCHS", 25))
LEARNING_RATE = 1e-3
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", 500))
SEED = 0
 
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
K = len(CLASSES)
 
# Each setting below is one design choice. The comment says why it is there
# and what it costs.
CONFIGS = {
    "baseline": dict(
        img_size=48,           # version 1 value
        pad_to_square=False,   # version 1 stretches every map into a square
        three_channels=False,  # version 1: one channel with values 0 / 0.5 / 1
        augment=False,         # version 1 had none
        weight_power=1.0,      # version 1: full inverse-frequency class weights
        optimizer="adam",      # version 1
        cosine_schedule=False, # version 1: constant learning rate
        batch_size=32,         # version 1
        patience=EPOCHS,       # version 1 always ran every epoch
        tta=False,             # version 1 predicted each wafer once
    ),
    "improved": dict(
        img_size=64,           # more pixels keep thin scratches visible. Cost: slower, more memory.
        pad_to_square=True,    # pad with "no die" instead of stretching, so shapes are not distorted. Cost: some empty border.
        three_channels=True,   # separate channels for outside / pass / fail, so "fail" is not "twice pass". Cost: slightly larger first layer.
        augment=True,          # random rotation and flip. Every defect label stays the same under these. Cost: slower to fit the training set.
        weight_power=0.5,      # square-root class weights: gentler than full inverse. Cost: rare classes get a weaker push.
        optimizer="adamw",     # Adam with weight decay, a small regularizer. Cost: one more setting (1e-4).
        cosine_schedule=True,  # learning rate shrinks smoothly to 0, which calms the noisy last epochs. Cost: need to know EPOCHS in advance.
        batch_size=64,         # fewer, steadier steps and faster epochs. Cost: this is one more thing that differs from baseline.
        patience=6,            # stop if val macro-F1 has not improved for 6 epochs. Cost: could stop a slow starter too early.
        tta=True,              # test-time augmentation: average predictions over 8 rotated/flipped views. Cost: 8x slower prediction.
    ),
}
 
os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
 
print("=" * 78)
print("CONFIG")
print("=" * 78)
print("device:", device, " epochs:", EPOCHS, " seed:", SEED, " bootstrap:", N_BOOTSTRAP)
for name, cfg in CONFIGS.items():
    print(name, cfg)



# =============================================================================
# 1. Load the pickle
# =============================================================================
print("\n" + "=" * 78)
print("1. LOAD DATA")
print("=" * 78)



pkl_candidates = glob.glob(os.path.join(DATA_DIR, "**", "*.pkl"), recursive=True)
print(f"Found {len(pkl_candidates)} .pkl file(s) under {DATA_DIR}")
assert len(pkl_candidates) > 0, f"No .pkl file found under {DATA_DIR}"
raw = pd.read_pickle(pkl_candidates[0])
print("Shape:", raw.shape)
 
# =============================================================================
# 2. Clean the labels and keep only labeled wafers
# =============================================================================
# failureType (and sometimes lotName) is stored as a nested array. unwrap_label
# pulls out the plain value, or None if it is empty.
print("\n" + "=" * 78)
print("2. LABELS AND CLASS BALANCE")
print("=" * 78)
 
 
def unwrap_label(value):
    while isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[0]
    return value
 
 
raw["failureType"] = raw["failureType"].apply(unwrap_label)
raw["lotName"] = raw["lotName"].apply(unwrap_label)
labeled = raw[raw["failureType"].isin(CLASSES)].reset_index(drop=True)
print(f"{len(labeled)} of {len(raw)} wafers carry a usable label.")
 
y = labeled["failureType"].map(CLASS_TO_IDX).to_numpy()
lots = labeled["lotName"].astype(str).to_numpy()
wafers = labeled["waferMap"].tolist()
del raw   # free memory, the full table is large
 
class_counts = np.bincount(y, minlength=K)
print(pd.DataFrame({"count": class_counts, "share_%": (100 * class_counts / len(y)).round(2)},
                   index=CLASSES))
 
# =============================================================================
# 3. Build the image arrays (one per distinct size/padding setting)
# =============================================================================
# Maps are stored as uint8 (values 0, 1, 2). That is 4x smaller than float32.
# They are turned into floats later, one batch at a time.
# Nearest-neighbour sampling keeps the values exactly 0, 1, 2.
print("\n" + "=" * 78)
print("3. RESIZE TO FIXED SIZE")
print("=" * 78)
 
 
def to_square_image(wafer, size, pad_to_square):
    w = np.asarray(wafer)
    h, wd = w.shape
    if pad_to_square:
        s = max(h, wd)
        canvas = np.zeros((s, s), dtype=np.uint8)          # 0 = no die
        r0, c0 = (s - h) // 2, (s - wd) // 2
        canvas[r0:r0 + h, c0:c0 + wd] = w
        w, h, wd = canvas, s, s
    rows = np.clip((np.arange(size) * h / size).astype(int), 0, h - 1)
    cols = np.clip((np.arange(size) * wd / size).astype(int), 0, wd - 1)
    return w[np.ix_(rows, cols)].astype(np.uint8)
 
 
image_cache = {}
for name, cfg in CONFIGS.items():
    key = (cfg["img_size"], cfg["pad_to_square"])
    if key not in image_cache:
        t0 = time.time()
        image_cache[key] = np.stack([to_square_image(w, *key) for w in wafers])
        print(f"size={key[0]} pad={key[1]}: array {image_cache[key].shape} "
              f"({image_cache[key].nbytes / 1e6:.0f} MB), {time.time() - t0:.0f}s, "
              f"values {np.unique(image_cache[key])}")
del wafers


# =============================================================================
# 4. Split by LOT (the main honesty fix)
# =============================================================================
# Version 1 split wafer by wafer, so one lot could sit in both train and test.
# Wafers of one lot share fab conditions, which made the test score optimistic.
# StratifiedGroupKFold keeps every lot in exactly one fold and still tries to
# keep class proportions. We use 10 folds: 1 for test, 1 for validation, 8 for
# train (80 / 10 / 10 as before).
# Cost: class proportions are only approximately equal across folds, and the
# score will probably be lower than in version 1. That lower number is the
# honest one.
print("\n" + "=" * 78)
print("4. SPLIT BY LOT")
print("=" * 78)
 
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
folds = [test_idx for _, test_idx in sgkf.split(np.zeros(len(y)), y, lots)]
idx_test, idx_val = folds[0], folds[1]
idx_train = np.concatenate(folds[2:])
 
overlap = set(lots[idx_train]) & set(lots[idx_test])
overlap_val = set(lots[idx_train]) & set(lots[idx_val])
print(f"train {len(idx_train)}  val {len(idx_val)}  test {len(idx_test)}")
print(f"Lots shared between train and test: {len(overlap)}, train and val: {len(overlap_val)}")
assert len(overlap) == 0 and len(overlap_val) == 0
 
split_table = pd.DataFrame({
    "train": np.bincount(y[idx_train], minlength=K),
    "val": np.bincount(y[idx_val], minlength=K),
    "test": np.bincount(y[idx_test], minlength=K),
}, index=CLASSES)
print(split_table)
split_table.to_csv(f"{OUT_DIR}/01_split_counts.csv")
if (split_table["test"] < 10).any():
    print("WARNING: some classes have fewer than 10 test wafers. "
          "Their F1 will be very noisy, read the confidence intervals.")
 
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
majority_acc = (y_test == CLASS_TO_IDX["none"]).mean()
print(f"Baseline 'always predict none': accuracy={majority_acc:.3f}")
 
