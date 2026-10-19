

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


# =============================================================================
# 5. Model and helper functions
# =============================================================================
# The CNN is the same as version 1 on purpose, so that any change in score
# comes from the data and training changes, not from a new architecture.
# Only the number of input channels differs (1 or 3).
 
 
class WaferCNN(nn.Module):
    def __init__(self, in_channels, num_classes=K):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
 
    def forward(self, x):
        return self.classifier(self.features(x))
 
 
def to_input(xb, three_channels):
    # xb: uint8 tensor (N, H, W) with values 0, 1, 2
    if three_channels:
        return torch.stack([(xb == 0), (xb == 1), (xb == 2)], dim=1).float()
    return (xb.float() / 2.0).unsqueeze(1)
 
 
def augment(x):
    # One random rotation (0/90/180/270) and one random flip per batch.
    # Same transform for the whole batch keeps the code simple. With thousands
    # of batches per epoch, every wafer still sees many different views.
    k = np.random.randint(4)
    x = torch.rot90(x, k, dims=(2, 3))
    if np.random.rand() < 0.5:
        x = torch.flip(x, dims=(3,))
    return x
 
 
def predict_probs(model, X_u8, cfg, tta):
    # Returns softmax probabilities, shape (N, K).
    model.eval()
    views = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    out = []
    with torch.no_grad():
        for start in range(0, len(X_u8), 512):
            xb = to_input(X_u8[start:start + 512].to(device), cfg["three_channels"])
            p = 0
            for k, f in views:
                v = torch.rot90(xb, k, dims=(2, 3))
                if f:
                    v = torch.flip(v, dims=(3,))
                p = p + model(v).softmax(dim=1)
            out.append((p / len(views)).cpu().numpy())
    return np.concatenate(out)
 
 
def run_experiment(name, cfg):
    print("\n" + "=" * 78)
    print(f"TRAINING: {name}")
    print("=" * 78)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
 
    X_all = torch.from_numpy(image_cache[(cfg["img_size"], cfg["pad_to_square"])])
    X_train, X_val, X_test = X_all[idx_train], X_all[idx_val], X_all[idx_test]
    yt_train = torch.from_numpy(y_train)
 
    # Class weights: (total / (K * count)) ** power. Power 1 = full inverse
    # frequency, power 0.5 = square root (gentler).
    counts = np.bincount(y_train, minlength=K).astype(np.float64)
    weights = (counts.sum() / (K * counts)) ** cfg["weight_power"]
    print("class weights:", {c: round(float(w), 2) for c, w in zip(CLASSES, weights)})
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
 
    in_ch = 3 if cfg["three_channels"] else 1
    model = WaferCNN(in_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {n_params:,}")
 
    if cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
                 if cfg["cosine_schedule"] else None)
 
    history = {"train_loss": [], "val_macro_f1": []}
    best_f1, best_epoch, best_state, bad_epochs = -1.0, 0, None, 0
    t_start = time.time()
 
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for start in range(0, len(perm), cfg["batch_size"]):
            idx = perm[start:start + cfg["batch_size"]]
            xb = to_input(X_train[idx].to(device), cfg["three_channels"])
            yb = yt_train[idx].to(device)
            if cfg["augment"]:
                xb = augment(xb)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if scheduler is not None:
            scheduler.step()
 
        # Validation never uses TTA, so epochs stay fast.
        val_pred = predict_probs(model, X_val, cfg, tta=False).argmax(1)
        val_f1 = f1_score(y_val, val_pred, average="macro", labels=range(K), zero_division=0)
        history["train_loss"].append(total_loss / len(perm))
        history["val_macro_f1"].append(val_f1)
        print(f"epoch {epoch:3d}/{EPOCHS} train_loss={history['train_loss'][-1]:.4f} "
              f"val_macroF1={val_f1:.3f}")
 
        if val_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = val_f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"early stop: no improvement for {cfg['patience']} epochs")
                break
 
    train_seconds = time.time() - t_start
    model.load_state_dict(best_state)
    print(f"best val macro-F1 {best_f1:.3f} at epoch {best_epoch}, "
          f"training time {train_seconds / 60:.1f} min")
 
    # Test set: used once, after the checkpoint is chosen.
    probs = predict_probs(model, X_test, cfg, tta=cfg["tta"])
    pd.DataFrame(history).rename_axis("epoch").to_csv(f"{OUT_DIR}/02_history_{name}.csv")
    torch.save(model.state_dict(), f"{OUT_DIR}/wafer_cnn_{name}.pt")
    return dict(probs=probs, pred=probs.argmax(1), best_val_f1=best_f1, best_epoch=best_epoch,
                train_minutes=train_seconds / 60, n_params=n_params, history=history)
 
 
results = {name: run_experiment(name, cfg) for name, cfg in CONFIGS.items()}


# =============================================================================
# 6. Compare the two models on the test set
# =============================================================================
print("\n" + "=" * 78)
print("6. TEST SET COMPARISON (same wafers, same lot-grouped split)")
print("=" * 78)
 
# Bootstrap: resample the test wafers with replacement N_BOOTSTRAP times and
# recompute macro-F1 each time. The spread of those numbers is the confidence
# interval. Both models use the SAME resamples, so the difference between them
# also gets an interval (a paired bootstrap).
# Cost: wafers from the same lot are resampled independently, so the intervals
# are slightly too narrow. Treat them as a guide.
rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, len(y_test), size=(N_BOOTSTRAP, len(y_test)))
 
 
def macro_f1(true, pred):
    return f1_score(true, pred, average="macro", labels=range(K), zero_division=0)
 
 
boot = {name: np.array([macro_f1(y_test[b], r["pred"][b]) for b in boot_idx])
        for name, r in results.items()}
 
rows = []
for name, r in results.items():
    rep = classification_report(y_test, r["pred"], labels=range(K), target_names=CLASSES,
                                output_dict=True, zero_division=0)
    lo, hi = np.percentile(boot[name], [2.5, 97.5])
    rows.append({
        "model": name,
        "test_accuracy": round(rep["accuracy"], 3),
        "test_macro_f1": round(rep["macro avg"]["f1-score"], 3),
        "macro_f1_95ci": f"{lo:.3f} to {hi:.3f}",
        "test_macro_recall": round(rep["macro avg"]["recall"], 3),
        "test_weighted_f1": round(rep["weighted avg"]["f1-score"], 3),
        "best_val_macro_f1": round(r["best_val_f1"], 3),
        "best_epoch": r["best_epoch"],
        "train_minutes": round(r["train_minutes"], 1),
        "parameters": r["n_params"],
    })
    r["report"] = rep
 
comparison = pd.DataFrame(rows).set_index("model")
print(comparison.T)
comparison.to_csv(f"{OUT_DIR}/03_comparison.csv")
 
diff = boot["improved"] - boot["baseline"]
d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
point_diff = comparison.loc["improved", "test_macro_f1"] - comparison.loc["baseline", "test_macro_f1"]
print(f"\nMacro-F1 change (improved - baseline): {point_diff:+.3f}, 95% interval {d_lo:+.3f} to {d_hi:+.3f}")
if d_lo > 0:
    verdict = "BETTER (interval is entirely above 0)"
elif d_hi < 0:
    verdict = "WORSE (interval is entirely below 0)"
else:
    verdict = "INCONCLUSIVE (interval includes 0)"
print("Verdict:", verdict)
 
per_class = pd.DataFrame({name: [results[name]["report"][c]["f1-score"] for c in CLASSES]
                          for name in results}, index=CLASSES)
per_class["change"] = per_class["improved"] - per_class["baseline"]
per_class["test_wafers"] = np.bincount(y_test, minlength=K)
print("\nPer-class test F1:")
print(per_class.round(3))
per_class.to_csv(f"{OUT_DIR}/04_per_class_f1.csv")
 
# =============================================================================
# 7. Figures
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 4))
per_class[["baseline", "improved"]].plot(kind="bar", ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("test F1")
ax.set_title("Per-class test F1: baseline vs improved")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_per_class_f1.png", dpi=150)
plt.close()
 
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, r in results.items():
    axes[0].plot(r["history"]["train_loss"], label=name)
    axes[1].plot(r["history"]["val_macro_f1"], label=name)
axes[0].set_title("Training loss (not comparable across models: different weights)")
axes[1].set_title("Validation macro-F1")
for a in axes:
    a.set_xlabel("epoch")
    a.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_training_curves.png", dpi=150)
plt.close()
 
for name, r in results.items():
    cm = confusion_matrix(y_test, r["pred"], labels=range(K)).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix, normalized by true class ({name})")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/07_confusion_{name}.png", dpi=150)
    plt.close()



# =============================================================================
# 8. Summary
# =============================================================================
summary = {
    "labeled_wafers": int(len(y)),
    "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
    "lots_shared_train_test": len(overlap),
    "baseline_always_none_accuracy": round(float(majority_acc), 3),
    "baseline_test_macro_f1": float(comparison.loc["baseline", "test_macro_f1"]),
    "improved_test_macro_f1": float(comparison.loc["improved", "test_macro_f1"]),
    "macro_f1_change": round(float(point_diff), 3),
    "macro_f1_change_95ci": [round(float(d_lo), 3), round(float(d_hi), 3)],
    "verdict": verdict,
}
with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
 
print("\n" + "=" * 78)
print("FINAL SUMMARY")
print("=" * 78)
for k, v in summary.items():
    print(f"{k:34s}: {v}")
print(f"\nSaved models, figures and tables to: {OUT_DIR}/")
print("This is one seed and one split. To check stability, change SEED at the top, "
      "run again, and compare the macro_f1_change values.")
 
