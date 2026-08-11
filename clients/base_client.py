# =============================================================================
# clients/base_client.py
# Flower FL client for a smart-grid substation, using the real NN model.
#
# Fix vs. the previous version: set_weights() here loads the aggregated
# global parameters directly into the PyTorch model before local training
# continues, so each round genuinely builds on the federated model instead
# of silently retraining an independent model from scratch.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import flwr as fl
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config.settings import PROCESSED_DIR, LOGS_DIR, MODELS_DIR, LOCAL_EPOCHS, FL_SERVER_ADDRESS
from models.nn_model import SmartGridModelWrapper, set_global_seed

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CLIENT_NAME_MAP = {"1": "client_A", "2": "client_B", "3": "client_C"}


def log_client(client_id: str, msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] [CLIENT-{client_id}] {msg}"
    print(line)
    with open(os.path.join(LOGS_DIR, f"client_{client_id}.log"), "a") as f:
        f.write(line + "\n")


def load_client_data(client_id: str):
    name = CLIENT_NAME_MAP.get(str(client_id), str(client_id))
    cdir = os.path.join(PROCESSED_DIR, name)
    train_path = os.path.join(cdir, "train.csv")
    test_path = os.path.join(cdir, "test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Client data not found at {train_path}.\n"
            "Run: python preprocessing/preprocess.py first "
            "(with the real CIC-IDS-2017 CSVs in dataset/)."
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feat_cols = [c for c in train_df.columns if c != "attack_class"]
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df["attack_class"].values.astype(np.int64)
    X_test = test_df[feat_cols].values.astype(np.float32)
    y_test = test_df["attack_class"].values.astype(np.int64)
    return X_train, y_train, X_test, y_test


class SmartGridClient(fl.client.NumPyClient):
    """
    Flower client for one substation.

    Privacy guarantee: X_train/X_test are loaded from this client's own
    CSV partition and never leave this process. Only NN weight tensors
    (get_weights/set_weights) cross the network via Flower.
    """

    def __init__(self, client_id: str):
        self.client_id = str(client_id)
        self.X_train, self.y_train, self.X_test, self.y_test = load_client_data(client_id)
        class_weights = SmartGridModelWrapper.compute_class_weights(self.y_train)
        self.model = SmartGridModelWrapper(client_id=f"client-{client_id}", class_weights=class_weights)
        self.round_num = 0
        log_client(self.client_id,
                    f"Data loaded | train={len(self.X_train):,} | test={len(self.X_test):,} | "
                    f"class_weights={class_weights}")

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.round_num += 1
        # Real fix: load the aggregated global weights before continuing
        # training, so the federated model actually accumulates knowledge
        # across rounds instead of each client training in isolation.
        self.model.set_weights(parameters)
        log_client(self.client_id, f"fit() round {self.round_num} — applied global weights")

        self.model.fit(self.X_train, self.y_train, epochs=LOCAL_EPOCHS, verbose=False)
        metrics = self.model.evaluate(self.X_test, self.y_test)

        log_client(self.client_id,
                   f"round {self.round_num} | acc={metrics['accuracy']} | "
                   f"f1={metrics['f1_weighted']}")

        return (
            self.model.get_weights(),
            len(self.X_train),
            {
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1_weighted"],
                "precision": metrics["precision_weighted"],
                "recall": metrics["recall_weighted"],
                "attack_precision": metrics.get("attack_precision", 0),
                "attack_recall": metrics.get("attack_recall", 0),
                "attack_f1": metrics.get("attack_f1", 0),
                "round": self.round_num,
                "client_id": self.client_id,
            },
        )

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        metrics = self.model.evaluate(self.X_test, self.y_test)
        loss = 1.0 - metrics["accuracy"]
        return (
            loss,
            len(self.X_test),
            {
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1_weighted"],
                "precision": metrics["precision_weighted"],
                "recall": metrics["recall_weighted"],
                "attack_precision": metrics.get("attack_precision", 0),
                "attack_recall": metrics.get("attack_recall", 0),
                "attack_f1": metrics.get("attack_f1", 0),
            },
        )


def start_client(client_id: str):
    set_global_seed(42)  # reproducibility — same init for all clients
    log_client(client_id, "[SEED] torch/numpy/random seeded with 42")
    log_client(client_id, f"Starting FL client {client_id}")
    client = SmartGridClient(client_id=client_id)
    fl.client.start_numpy_client(server_address=FL_SERVER_ADDRESS, client=client)
    log_client(client_id, "FL training complete.")
