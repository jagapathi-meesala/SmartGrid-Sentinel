# =============================================================================
# train_baseline.py
# Trains the centralized NN baseline on the full processed CIC-IDS-2017
# dataset and reports real accuracy / F1 / confusion matrix — the number
# that appears in the report's Results section should come from running
# this script, not from a placeholder.
#
# Run: python train_baseline.py   (after preprocessing/preprocess.py)
# =============================================================================

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import PROCESSED_DIR, RESULTS_DIR, MODELS_DIR, NN_EPOCHS, CLASS_NAMES
from models.nn_model import SmartGridModelWrapper, set_global_seed

SPLIT_PATH = os.path.join(PROCESSED_DIR, "centralized_split.npz")


def main():
    set_global_seed(42)  # reproducibility — must be first
    print("[SEED] torch/numpy/random seeded with 42 for reproducibility")
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(
            f"{SPLIT_PATH} not found. Run preprocessing/preprocess.py first "
            "with the real CIC-IDS-2017 CSVs in dataset/."
        )

    data = np.load(SPLIT_PATH)
    X_train, X_test = data["X_train"].astype(np.float32), data["X_test"].astype(np.float32)
    y_train, y_test = data["y_train"].astype(np.int64), data["y_test"].astype(np.int64)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    class_weights = SmartGridModelWrapper.compute_class_weights(y_train)
    print(f"Class weights (inverse-frequency): {class_weights}")
    model = SmartGridModelWrapper(client_id="centralized-baseline", class_weights=class_weights)

    # Train epoch-by-epoch so we can plot real loss/accuracy curves
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(NN_EPOCHS):
        h = model.fit(X_train, y_train, epochs=1, verbose=True)
        history["train_loss"].append(h["loss"][0])
        history["train_acc"].append(h["acc"][0])

        val_metrics = model.evaluate(X_test, y_test)
        history["val_acc"].append(val_metrics["accuracy"])
        # approximate val loss via cross-entropy on held-out set
        val_probs = model.predict_proba(X_test)
        val_loss = -np.mean(np.log(val_probs[np.arange(len(y_test)), y_test] + 1e-9))
        history["val_loss"].append(float(val_loss))
        print(f"Epoch {epoch+1}/{NN_EPOCHS} | val_acc={val_metrics['accuracy']:.4f} "
              f"| val_f1={val_metrics['f1_weighted']:.4f}")

    final_metrics = model.evaluate(X_test, y_test)
    print("\n=== FINAL CENTRALIZED BASELINE (Optimized Architecture & Loss) ===")
    print(f"Accuracy         : {final_metrics['accuracy'] * 100:.2f}%")
    print(f"Attack Precision : {final_metrics['attack_precision'] * 100:.2f}%")
    print(f"Attack Recall    : {final_metrics['attack_recall'] * 100:.2f}%")
    print(f"Attack F1        : {final_metrics['attack_f1'] * 100:.2f}%")
    print(f"F1 (weighted)    : {final_metrics['f1_weighted']}")
    print(f"Precision        : {final_metrics['precision_weighted']}")
    print(f"Recall           : {final_metrics['recall_weighted']}")
    print(f"Per-class recall : {dict(zip(CLASS_NAMES.values(), final_metrics['per_class_recall']))}")

    # Save model checkpoint
    ckpt_path = os.path.join(MODELS_DIR, "centralized_baseline.pt")
    model.save(ckpt_path)
    print(f"Saved model -> {ckpt_path}")

    # Save metrics + history for the paper's Results section
    out = {
        "final_metrics": final_metrics,
        "history": history,
        "class_names": CLASS_NAMES,
    }
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Real loss/accuracy curve plots (replacing any hardcoded figure)
    fig, ax = plt.subplots()
    ax.plot(range(1, NN_EPOCHS+1), history["train_loss"], label="Train")
    ax.plot(range(1, NN_EPOCHS+1), history["val_loss"], label="Val")
    ax.set_title("Loss"); ax.legend()
    fig.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))

    fig, ax = plt.subplots()
    ax.plot(range(1, NN_EPOCHS+1), [a*100 for a in history["train_acc"]], label="Train")
    ax.plot(range(1, NN_EPOCHS+1), [a*100 for a in history["val_acc"]], label="Val")
    ax.set_title("Accuracy %"); ax.legend()
    fig.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"))

    print(f"Saved plots + metrics -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
