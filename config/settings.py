# =============================================================================
# config/settings.py
# Central configuration for the SmartGrid Sentinel FL project.
# Matches the architecture and dataset described in the project report:
#   - CIC-IDS-2017 traffic dataset (NOT NSL-KDD)
#   - 77 numerical features, 15 traffic classes
#   - 3-layer NN: 77 -> Dense(64) -> Dense(32) -> Dense(15) softmax
# =============================================================================

import os

# ── Project Root ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")
MODELS_DIR    = os.path.join(BASE_DIR, "models", "saved")

for _dir in [DATASET_DIR, PROCESSED_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Dataset: CIC-IDS-2017 (used for the digest — do not change) ─────────────
# The 8 official daily CSVs from the University of New Brunswick's CIC-IDS-2017
# release (~844 MB total). This project does NOT auto-download them: the
# official distribution requires a request form, and no automated mirror is
# used here to avoid distributing the dataset outside its license terms.
CIC_DATASET_DIR = os.path.join(DATASET_DIR, "cicids2017")

# ── Dataset: HAI (used for the full paper) ───────────────────────────────────
# HIL-based Augmented ICS Security Dataset (github.com/icsdataset/hai).
# Real physical-process sensor data from a testbed emulating steam-turbine
# and pumped-storage power generation (boiler P1 / turbine P2 / water
# treatment P3, coupled via a HIL simulator P4). Freely downloadable, no
# license-request wait — see preprocessing/preprocess_hai.py.
#
# IMPORTANT DESIGN NOTE: HAI's official train1/2/3.csv files are pure-normal
# operation only (designed for unsupervised/semi-supervised anomaly
# detection). To keep this project's *supervised* classification approach
# unchanged, we instead use the 5 official *test* files (test1..test5.csv),
# which do contain real labeled attack windows, as our labeled corpus, and
# perform our own stratified train/test split and client partition on them.
HAI_DIR = os.path.join(DATASET_DIR, "hai-21.03")
HAI_LABELED_FILES = ["test1.csv", "test2.csv", "test3.csv", "test4.csv", "test5.csv"]
# Non-IID client assignment: each client gets whole, distinct source files
# (real separate experiment runs), not a random pooled split.
HAI_CLIENT_FILE_MAP = {
    "client_A": ["test1.csv", "test2.csv"],   # ~162k rows
    "client_B": ["test3.csv"],                 # ~108k rows
    "client_C": ["test4.csv", "test5.csv"],   # ~132k rows
}
HAI_LABEL_COLS = ["attack", "attack_P1", "attack_P2", "attack_P3"]
HAI_DROP_COLS = ["time"]

# ── Active dataset switch ─────────────────────────────────────────────────────
# "HAI" for the full paper (current default), "CICIDS" for the digest.
ACTIVE_DATASET = "HAI"

NUM_CLIENTS = 3   # Substation A, B, C in both datasets

if ACTIVE_DATASET == "HAI":
    NUM_FEATURES = 79
    NUM_CLASSES = 2
    CLASS_NAMES = {0: "Normal", 1: "Attack"}
    NORMAL_LABEL_NAME = "Normal"
else:
    # 15 traffic classes exactly as in the report / CIC-IDS-2017 official labels
    CIC_LABEL_MAP = {
        "BENIGN": 0,
        "DDoS": 1,
        "DoS Hulk": 2,
        "DoS GoldenEye": 3,
        "DoS slowloris": 4,
        "DoS Slowhttptest": 5,
        "PortScan": 6,
        "FTP-Patator": 7,
        "SSH-Patator": 8,
        "Bot": 9,
        "Web Attack \x96 Brute Force": 10,
        "Web Attack - Brute Force": 10,
        "Web Attack \x96 XSS": 11,
        "Web Attack - XSS": 11,
        "Web Attack \x96 Sql Injection": 12,
        "Web Attack - Sql Injection": 12,
        "Infiltration": 13,
        "Heartbleed": 14,
    }
    CLASS_NAMES = {
        0: "BENIGN", 1: "DDoS", 2: "DoS Hulk", 3: "DoS GoldenEye",
        4: "DoS Slowloris", 5: "DoS Slowhttptest", 6: "PortScan",
        7: "FTP-Patator", 8: "SSH-Patator", 9: "Bot",
        10: "Web Attack - Brute Force", 11: "Web Attack - XSS",
        12: "Web Attack - SQL Injection", 13: "Infiltration", 14: "Heartbleed",
    }
    NUM_FEATURES = 77
    NUM_CLASSES = 15
    NORMAL_LABEL_NAME = "BENIGN"
    # Raw CIC-IDS-2017 CSVs have 78 feature columns + Label = 79 columns/row.
    # One column ("Fwd Header Length.1") is an exact duplicate and is dropped,
    # giving the 77 unique numerical features the report trains on.
    DROP_DUPLICATE_COLS = ["Fwd Header Length.1"]
    NON_FEATURE_COLS = ["Label"]

# ── Federated Learning ────────────────────────────────────────────────────────
FL_SERVER_ADDRESS  = "127.0.0.1:8080"
FL_ROUNDS          = 10     # communication rounds — matches report
FL_MIN_CLIENTS     = 3
FL_MIN_AVAILABLE   = 3
LOCAL_EPOCHS       = 3      # local epochs per client per round — matches report
LOCAL_BATCH_SIZE   = 1024   # vectorized batch size for fast CPU training

# ── Model: Enhanced Multi-layer NN ───────────────────────────────────────────
NN_HIDDEN_LAYERS = [128, 64, 32]
NN_DROPOUT       = 0.2
NN_LR            = 2e-3
NN_WEIGHT_DECAY  = 1e-4
NN_EPOCHS        = 6        # centralized baseline epochs for convergence

# ── Digital Twin ──────────────────────────────────────────────────────────────
DT_NUM_TICKS       = 30
DT_FLOWS_PER_TICK  = 15     # per substation -> 45/tick, 1350 total (matches report)
DT_RISK_K          = 0.3    # dynamic threshold = mean(conf) + K * std(conf)
if ACTIVE_DATASET == "HAI":
    DT_INJECTED_EVENTS = [
        {"tick": 10, "substation": "SUB-A", "attack": "Attack"},
        {"tick": 15, "substation": "SUB-C", "attack": "Attack"},
        {"tick": 25, "substation": "SUB-A", "attack": "Attack"},
    ]
else:
    DT_INJECTED_EVENTS = [
        {"tick": 10, "substation": "SUB-A", "attack": "DDoS"},
        {"tick": 15, "substation": "SUB-C", "attack": "PortScan"},
        {"tick": 25, "substation": "SUB-A", "attack": "DDoS"},
    ]
SUBSTATIONS = ["SUB-A", "SUB-B", "SUB-C"]

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_PORT     = 8501
DASHBOARD_REFRESH  = 3
