

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
