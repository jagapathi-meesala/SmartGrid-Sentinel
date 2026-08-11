# =============================================================================
# preprocessing/preprocess.py
# Loads the 8 official CIC-IDS-2017 CSVs, cleans them, encodes the 15 classes,
# scales features, splits 80/20, and partitions into 3 client datasets.
#
# This replaces the previous NSL-KDD-based script. It does NOT download the
# dataset automatically — CIC-IDS-2017 requires accepting UNB's license terms
# via their request form. Place the 8 official CSVs in DATASET_DIR first.
#
# Run: python preprocessing/preprocess.py
# =============================================================================

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    DATASET_DIR, PROCESSED_DIR, MODELS_DIR, CIC_LABEL_MAP,
    DROP_DUPLICATE_COLS, NON_FEATURE_COLS, NUM_CLIENTS, NUM_FEATURES,
)

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_columns.txt")


def load_raw() -> pd.DataFrame:
    """Load and concatenate all CIC-IDS-2017 daily CSVs found in DATASET_DIR."""
    csv_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSVs found in {DATASET_DIR}.\n"
            "Download the 8 official CIC-IDS-2017 CSVs from the UNB CIC "
            "distribution (request form required) and place them, unmodified, "
            f"in {DATASET_DIR} before running this script."
        )
    print(f"[LOAD] Found {len(csv_paths)} CSV file(s):")
    frames = []
    for p in csv_paths:
        print(f"  - {os.path.basename(p)}")
        # CIC-IDS-2017 CSVs are large; low_memory=False avoids dtype-inference
        # warnings from mixed chunks. encoding='latin1' handles the raw
        # dataset's non-UTF8 dash character in some Web Attack labels.
        df = pd.read_csv(p, low_memory=False, encoding="latin1")
        df.columns = df.columns.str.strip()
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    print(f"[LOAD] Combined: {len(full):,} rows x {len(full.columns)} cols")
    return full


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the report's exact cleaning pipeline, step by step, with counts."""
    df = df.copy()
    n0 = len(df)
    print(f"\n[STEP 1] Starting rows: {n0:,}")

    # Drop duplicate engineered column present in the official release
    for col in DROP_DUPLICATE_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Replace inf with NaN so both are handled by the same dropna pass counts
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    n_before_nan = len(df)
    df.dropna(inplace=True)
    print(f"[STEP 2] Removed NaN/inf rows -> {len(df):,} "
          f"({n_before_nan - len(df):,} removed)")

    n_before_dup = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[STEP 3] Removed exact duplicate rows -> {len(df):,} "
          f"({n_before_dup - len(df):,} removed)")

    print("[STEP 4] Encoding labels to 15 integer classes")
    df["Label"] = df["Label"].astype(str).str.strip()
    df["attack_class"] = df["Label"].map(CIC_LABEL_MAP)
    unmapped = df["attack_class"].isna().sum()
    if unmapped > 0:
        unmapped_labels = df.loc[df["attack_class"].isna(), "Label"].unique()
        print(f"  [WARN] {unmapped:,} rows had unmapped labels: {unmapped_labels[:10]}")
        df = df[df["attack_class"].notna()]
    df["attack_class"] = df["attack_class"].astype(int)

    print(f"[STEP 5] Final cleaned dataset: {len(df):,} rows "
          f"({n0 - len(df):,} total removed from raw)")
    dist = df["attack_class"].value_counts().sort_index()
    print(f"  Class distribution:\n{dist.to_string()}")
    return df


def scale_and_split(df: pd.DataFrame):
    """Standardize the 77 numeric features and stratified 80/20 split."""
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS + ["attack_class"]]

    if len(feat_cols) != NUM_FEATURES:
        print(f"  [WARN] Found {len(feat_cols)} feature columns, expected "
              f"{NUM_FEATURES}. Check DROP_DUPLICATE_COLS / raw CSV schema.")

    X = df[feat_cols].astype(np.float64).values
    y = df["attack_class"].values

    print(f"\n[STEP 6] Scaling {X.shape[1]} features with StandardScaler")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_LIST_PATH, "w") as f:
        f.write("\n".join(feat_cols))
    print(f"  Saved scaler -> {SCALER_PATH}")
    print(f"  Saved feature column order -> {FEATURE_LIST_PATH}")

    print("\n[STEP 7] Stratified 80/20 train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    scaled_df = pd.DataFrame(X_scaled, columns=feat_cols)
    scaled_df["attack_class"] = y
    return scaled_df, (X_train, X_test, y_train, y_test), feat_cols


def split_for_clients(df: pd.DataFrame, n: int = NUM_CLIENTS):
    """IID partition into n client datasets (Substation A/B/C).

    NOTE: this is an IID split for reproducing the report's baseline FL
    numbers. It does not model realistic per-substation traffic
    heterogeneity — see README limitations section for a non-IID variant.
    """
    print(f"\n[STEP 8] Partitioning into {n} client datasets (IID)")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    chunk = len(df) // n
    client_names = ["client_A", "client_B", "client_C"][:n]
    for i, name in enumerate(client_names):
        start = i * chunk
        end = start + chunk if i < n - 1 else len(df)
        part = df.iloc[start:end].copy()
        tr, te = train_test_split(
            part, test_size=0.2, random_state=42, stratify=part["attack_class"]
        )
        cdir = os.path.join(PROCESSED_DIR, name)
        os.makedirs(cdir, exist_ok=True)
        tr.to_csv(os.path.join(cdir, "train.csv"), index=False)
        te.to_csv(os.path.join(cdir, "test.csv"), index=False)
        print(f"  {name}: train={len(tr):,} | test={len(te):,}")


def main():
    print("=" * 70)
    print("  SMARTGRID SENTINEL — CIC-IDS-2017 PREPROCESSING")
    print("=" * 70)
    raw = load_raw()
    cleaned = clean(raw)
    scaled_df, splits, feat_cols = scale_and_split(cleaned)
    scaled_df.to_csv(os.path.join(PROCESSED_DIR, "full_processed.csv"), index=False)
    split_for_clients(scaled_df)

    X_train, X_test, y_train, y_test = splits
    np.savez(
        os.path.join(PROCESSED_DIR, "centralized_split.npz"),
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    )
    print("\n[DONE] Preprocessing complete.")
    print(f"  Full data          : {PROCESSED_DIR}/full_processed.csv")
    print(f"  Centralized split  : {PROCESSED_DIR}/centralized_split.npz")
    print(f"  Client partitions  : {PROCESSED_DIR}/client_A|B|C/")


if __name__ == "__main__":
    main()
