# =============================================================================
# dashboard/api.py
# FastAPI backend for the SmartGrid TypeScript dashboard.
# Serves REAL data from results/ and logs/ directories over HTTP.
# Run: uvicorn dashboard.api:app --reload --port 8000
# =============================================================================

import os, json, re
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR     = Path(__file__).parent.parent
RESULTS_DIR  = BASE_DIR / "results"
LOGS_DIR     = BASE_DIR / "logs"

app = FastAPI(title="SmartGrid FL API", version="1.0.0")

# Allow Vite dev server (port 5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _tail_jsonl(path: Path, n: int):
    """Return last n records from a .jsonl file."""
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows[-n:]


def _parse_log_for_client(log_path: Path, client_id: str):
    """
    Extract the most recent accuracy and last-seen timestamp from a client log.
    Log lines look like:
      [2026-08-07T18:20:42Z] [CLIENT] ... acc=0.9865 f1=0.9852 ...
    """
    if not log_path.exists():
        return {"accuracy": None, "f1": None, "last_seen": None}

    acc = f1 = last_seen = None
    acc_re = re.compile(r"acc=([\d.]+)")
    f1_re  = re.compile(r"f1=([\d.]+)")
    ts_re  = re.compile(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?)\]")

    with open(log_path) as fh:
        for line in fh:
            m_ts  = ts_re.search(line)
            m_acc = acc_re.search(line)
            m_f1  = f1_re.search(line)
            if m_ts:
                last_seen = m_ts.group(1)
            if m_acc:
                acc = float(m_acc.group(1))
            if m_f1:
                f1 = float(m_f1.group(1))

    return {"accuracy": acc, "f1": f1, "last_seen": last_seen}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/fl-metrics")
def fl_metrics():
    """FL per-round accuracy & F1 from results/fl_metrics.json."""
    data = _read_json(RESULTS_DIR / "fl_metrics.json")
    if data is None:
        return {"rounds": []}
    return data


@app.get("/api/baseline")
def baseline():
    """Centralised baseline metrics from results/baseline_metrics.json."""
    data = _read_json(RESULTS_DIR / "baseline_metrics.json")
    if data is None:
        return {}
    return data


@app.get("/api/twin-events")
def twin_events(limit: int = Query(default=2000, le=5000)):
    """Last N digital-twin traffic events from results/twin_events.json."""
    data = _read_json(RESULTS_DIR / "twin_events.json")
    if data is None:
        return []
    if isinstance(data, list):
        return data[-limit:]
    # Some versions wrap in a key
    for key in ("events", "data", "records"):
        if key in data:
            return data[key][-limit:]
    return []


@app.get("/api/node-status")
def node_status():
    """
    Per-node status parsed from real log files.
    Returns last-seen timestamp and latest accuracy for each client.
    """
    nodes = [
        {"id": "SUB-01", "location": "North Grid",   "type": "Transmission", "log": "client_1.log"},
        {"id": "SUB-02", "location": "South Grid",   "type": "Distribution",  "log": "client_2.log"},
        {"id": "SUB-03", "location": "East Subunit", "type": "Generation",    "log": "client_3.log"},
    ]

    result = []
    for node in nodes:
        log_path = LOGS_DIR / node["log"]
        parsed   = _parse_log_for_client(log_path, node["id"])

        # Determine online/alert based on how recent last_seen is
        status = "unknown"
        if parsed["last_seen"]:
            try:
                ts_str = parsed["last_seen"].replace("Z", "+00:00")
                last_dt = datetime.fromisoformat(ts_str)
                age_hours = (datetime.utcnow() - last_dt.replace(tzinfo=None)).total_seconds() / 3600
                status = "online" if age_hours < 24 else "offline"
            except Exception:
                status = "online"  # log exists, assume active

        result.append({
            "id":        node["id"],
            "location":  node["location"],
            "type":      node["type"],
            "status":    status,
            "accuracy":  parsed["accuracy"],
            "f1":        parsed["f1"],
            "last_seen": parsed["last_seen"],
        })

    # Server log for fl rounds count
    server_log = LOGS_DIR / "server.log"
    fl_rounds_done = 0
    if server_log.exists():
        with open(server_log) as f:
            for line in f:
                if "Round" in line and "aggregation complete" in line.lower():
                    fl_rounds_done += 1

    return {"nodes": result, "fl_rounds_done": fl_rounds_done}


@app.get("/api/threat-intel")
def threat_intel():
    """Returns Threat Intelligence DB indicators and feed logs."""
    data = _read_json(RESULTS_DIR / "threat_db.json")
    if data is None:
        return {"total_iocs": 0, "indicators": []}
    return data


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

