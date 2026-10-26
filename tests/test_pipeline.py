"""Smoke tests for dengai_pipeline.py.

The tests build small fake data with the same columns as the DengAI files,
run the whole pipeline once in QUICK mode, and check the outputs.
They test that the pipeline runs and produces sane files.
They do not test how accurate the models are.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "dengai_pipeline.py"
CITIES = {"sj": 300, "iq": 220}   # number of training weeks per city
N_TEST_WEEKS = 52

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


def make_fake_data(data_dir):
    rng = np.random.default_rng(0)
    feat_train, labels, feat_test = [], [], []
    for city, n_train in CITIES.items():
        dates = pd.date_range("2000-01-02", periods=n_train + N_TEST_WEEKS, freq="7D")
        iso = dates.isocalendar()
        base = pd.DataFrame({
            "city": city,
            "year": iso["year"].values,
            "weekofyear": iso["week"].values,
            "week_start_date": dates,
        })
        for col in CLIMATE_COLUMNS:
            base[col] = rng.normal(20, 3, len(base))
        # a few missing values, like the real data has
        for col in CLIMATE_COLUMNS[:5]:
            base.loc[rng.choice(len(base), 5, replace=False), col] = np.nan
        base["total_cases"] = rng.poisson(15, len(base))

        train = base.iloc[:n_train]
        test = base.iloc[n_train:]
        feat_train.append(train.drop(columns="total_cases"))
        labels.append(train[["city", "year", "weekofyear", "total_cases"]])
        feat_test.append(test.drop(columns="total_cases"))

    pd.concat(feat_train).to_csv(data_dir / "dengue_features_train.csv", index=False)
    pd.concat(labels).to_csv(data_dir / "dengue_labels_train.csv", index=False)
    pd.concat(feat_test).to_csv(data_dir / "dengue_features_test.csv", index=False)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """Runs the pipeline once. All tests below read its outputs."""
    work = tmp_path_factory.mktemp("run")
    data_dir = work / "data"
    data_dir.mkdir()
    make_fake_data(data_dir)

    env = dict(os.environ, DATA_DIR=str(data_dir), QUICK="1")
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=work, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"pipeline failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return work


def test_submission_is_valid(run_dir):
    sub = pd.read_csv(run_dir / "submission.csv")
    assert list(sub.columns) == ["city", "year", "weekofyear", "total_cases"]
    assert len(sub) == N_TEST_WEEKS * len(CITIES)
    assert sub["total_cases"].notna().all()
    assert (sub["total_cases"] >= 0).all()


def test_version_is_registered(run_dir):
    registry = json.loads((run_dir / "models" / "registry.json").read_text())
    assert len(registry) == 1
    latest = registry[-1]
    assert Path(run_dir / latest["path"]).is_dir()
    assert set(latest["cv_mae"]) == set(CITIES)


def test_manifest_records_provenance(run_dir):
    registry = json.loads((run_dir / "models" / "registry.json").read_text())
    version_dir = run_dir / registry[-1]["path"]
    manifest = json.loads((version_dir / "manifest.json").read_text())
    assert manifest["version"] == registry[-1]["version"]
    assert "git_commit" in manifest
    assert len(manifest["data_sha256"]["features_train"]) == 64
    assert "lightgbm" in manifest["library_versions"]


def test_saved_models_load(run_dir):
    registry = json.loads((run_dir / "models" / "registry.json").read_text())
    version_dir = run_dir / registry[-1]["path"]
    for city in CITIES:
        bundle = joblib.load(version_dir / f"{city}_model.joblib")
        assert set(bundle) == {"model_kind", "model", "feature_columns"}
        assert len(bundle["feature_columns"]) > 0


def test_second_run_adds_a_new_version(run_dir):
    """Running again must not overwrite the first version."""
    env = dict(os.environ, DATA_DIR=str(run_dir / "data"), QUICK="1")
    import time
    time.sleep(1.1)  # version ids contain seconds, so wait to get a different one
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=run_dir, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    registry = json.loads((run_dir / "models" / "registry.json").read_text())
    assert len(registry) == 2
    assert registry[0]["version"] != registry[1]["version"]
    assert Path(run_dir / registry[0]["path"]).is_dir()
