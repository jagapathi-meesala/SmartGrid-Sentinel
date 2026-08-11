# =============================================================================
# preprocessing/preprocess_hai.py
# Preprocessing for the HAI (HIL-based Augmented ICS) Security Dataset.
#
# Produces exactly the same output schema as preprocessing/preprocess.py
# (feature columns + a single "attack_class" integer label column, per-client
# train.csv/test.csv, centralized_split.npz) so that every downstream script
# — models/nn_model.py, clients/base_client.py, server/server.py,
# train_baseline.py, digital_twin/simulator.py, dashboard/app.py — works
# completely unchanged. Only the data source and class count differ.
#
# Design note (read this before reporting results): HAI's official
# train1/2/3.csv are pure-normal-operation files (the dataset is designed
# for unsupervised/semi-supervised anomaly detection). To keep this
# project's supervised-classification approach unchanged, we use the 5
# official *test* files (which contain real labeled attack windows) as our
# labeled corpus, and do our own stratified train/test split. State this
# explicitly in the paper's methodology section — it's a deliberate,
# defensible adaptation, not something to gloss over.
#
# Run: python preprocessing/preprocess_hai.py
#   (after placing hai-21.03/test1..5.csv in dataset/hai-21.03/)
# =============================================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    HAI_DIR, HAI_LABELED_FILES, HAI_CLIENT_FILE_MAP, HAI_LABEL_COLS, HAI_DROP_COLS,
    PROCESSED_DIR, MODELS_DIR, NUM_FEATURES,
)

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_columns.txt")


def load_file(name: str) -> pd.DataFrame:
    path = os.path.join(HAI_DIR, name)
    if not os.path.exists(path):
        # allow gzipped originals too
        gz = path + ".gz"
        if os.path.exists(gz):
            path = gz
        else:
            raise FileNotFoundError(
                f"{path} not found. Download hai-21.03 from "
                "https://github.com/icsdataset/hai (test1.csv..test5.csv) "
                f"and place them in {HAI_DIR}/"
            )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)
    for c in HAI_DROP_COLS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    feat_cols = [c for c in df.columns if c not in HAI_LABEL_COLS]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Collapse to the same single-label schema as the CIC pipeline: one
    # integer column "attack_class" (0=Normal, 1=Attack). Per-subsystem
    # attribution (attack_P1/P2/P3) is kept as extra columns for optional
    # diagnostic reporting but is not the primary training target here.
    df["attack_class"] = df["attack"].astype(int)
    print(f"[CLEAN] {n0:,} -> {len(df):,} rows after NaN/inf/duplicate removal")
    return df


def build_labeled_pool() -> dict:
    """Load each of the 5 labeled test files separately (kept apart so we can
    do a non-IID, real-source-based client split rather than a random pool)."""
    pool = {}
    for fname in HAI_LABELED_FILES:
        print(f"[LOAD] {fname}")
        df = load_file(fname)
        df = clean(df)
        pool[fname] = df
        dist = df["attack_class"].value_counts().to_dict()
        print(f"  rows={len(df):,} | class dist={dist}")
    return pool


def main():
    print("=" * 70)
    print("  SMARTGRID SENTINEL — HAI PREPROCESSING")
    print("=" * 70)
    os.makedirs(HAI_DIR, exist_ok=True)

    pool = build_labeled_pool()
    feat_cols = [c for c in next(iter(pool.values())).columns
                 if c not in HAI_LABEL_COLS + ["attack_class"]]
    if len(feat_cols) != NUM_FEATURES:
        print(f"  [WARN] Found {len(feat_cols)} feature columns, expected {NUM_FEATURES}.")

    # Fit the scaler on the full pooled labeled corpus (fair, matches the
    # CIC pipeline's approach of fitting on the full cleaned dataset)
    full = pd.concat(pool.values(), ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(full[feat_cols].values.astype(np.float64))
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_LIST_PATH, "w") as f:
        f.write("\n".join(feat_cols))
    print(f"\n[SCALE] Fit StandardScaler on {len(full):,} pooled rows -> {SCALER_PATH}")

    # ── Per-client partition: real distinct source files, not a random split ──
    print("\n[SPLIT] Building non-IID client partitions from real source files")
    for client_name, files in HAI_CLIENT_FILE_MAP.items():
        client_df = pd.concat([pool[f] for f in files], ignore_index=True)
        X = scaler.transform(client_df[feat_cols].values.astype(np.float64))
        scaled_df = pd.DataFrame(X, columns=feat_cols)
        scaled_df["attack_class"] = client_df["attack_class"].values

        tr, te = train_test_split(
            scaled_df, test_size=0.2, random_state=42,
            stratify=scaled_df["attack_class"],
        )
        cdir = os.path.join(PROCESSED_DIR, client_name)
        os.makedirs(cdir, exist_ok=True)
        tr.to_csv(os.path.join(cdir, "train.csv"), index=False)
        te.to_csv(os.path.join(cdir, "test.csv"), index=False)
        print(f"  {client_name} (from {files}): train={len(tr):,} | test={len(te):,} | "
              f"attack%={scaled_df['attack_class'].mean()*100:.2f}%")

    # ── Centralized pool: all 5 files combined, standard stratified split ────
    print("\n[SPLIT] Building centralized (non-federated baseline) split")
    X_full = scaler.transform(full[feat_cols].values.astype(np.float64))
    y_full = full["attack_class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full,
    )
    np.savez(
        os.path.join(PROCESSED_DIR, "centralized_split.npz"),
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    )
    print(f"  Centralized: train={len(X_train):,} | test={len(X_test):,}")

    # Full processed pool (used by the digital twin to sample real flows)
    full_scaled = pd.DataFrame(X_full, columns=feat_cols)
    full_scaled["attack_class"] = y_full
    full_scaled.to_csv(os.path.join(PROCESSED_DIR, "full_processed.csv"), index=False)

    print("\n[DONE] HAI preprocessing complete.")
    print(f"  Full data          : {PROCESSED_DIR}/full_processed.csv")
    print(f"  Centralized split  : {PROCESSED_DIR}/centralized_split.npz")
    print(f"  Client partitions  : {PROCESSED_DIR}/client_A|B|C/")


if __name__ == "__main__":
    main()
