#!/usr/bin/env bash
# =============================================================================
# run_full_experiment.sh
# Single, clean, uninterrupted full experiment run.
# Seed 42 is baked into every Python module; this script just orchestrates.
#
# Usage: bash run_full_experiment.sh
# =============================================================================

set -e  # exit immediately on error
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"

echo "============================================================"
echo " SmartGrid FL — Full Experiment Run (seed=42)"
echo "============================================================"

# ── Step 1: Centralized baseline ─────────────────────────────────────────────
echo ""
echo "[1/5] Training centralized baseline (train_baseline.py)..."
$PYTHON train_baseline.py
echo "[1/5] DONE"

# ── Step 2: FL training — server + 3 clients (all 10 rounds) ─────────────────
echo ""
echo "[2/5] Federated Learning — starting server + 3 clients (10 rounds)..."

# Clear old server log so we can tail cleanly
> logs/server.log

# Start server in background
$PYTHON server/server.py &
SERVER_PID=$!
echo "      server PID=$SERVER_PID"
sleep 3   # give server a moment to bind the port

# Start all 3 clients in background
$PYTHON clients/client1.py &
C1_PID=$!
$PYTHON clients/client2.py &
C2_PID=$!
$PYTHON clients/client3.py &
C3_PID=$!
echo "      client PIDs: $C1_PID $C2_PID $C3_PID"

# Wait for all clients to finish (they exit after the server closes)
wait $C1_PID $C2_PID $C3_PID
echo "      All clients finished."
wait $SERVER_PID
echo "[2/5] DONE — fl_metrics.json written"

# ── Step 3: Digital Twin simulation ──────────────────────────────────────────
echo ""
echo "[3/5] Running Digital Twin simulation (digital_twin/simulator.py)..."
$PYTHON digital_twin/simulator.py
echo "[3/5] DONE — twin_events.json written"

# ── Step 4: Autonomous Response Agent ────────────────────────────────────────
echo ""
echo "[4/6] Running ARA agent (ara/agent.py)..."
$PYTHON ara/agent.py
echo "[4/6] DONE — ara_actions.json written"

# ── Step 5: Threat Intelligence Integration Agent ────────────────────────────
echo ""
echo "[5/6] Running Threat Intelligence Agent (threat_intel/agent.py)..."
$PYTHON threat_intel/agent.py
echo "[5/6] DONE — threat_intelligence.db & threat_db.json written"

# ── Step 6: Generate paper figures ───────────────────────────────────────────
echo ""
echo "[6/6] Generating paper figures (scripts/generate_paper_figures.py)..."
$PYTHON scripts/generate_paper_figures.py
echo "[5/5] DONE"

echo ""
echo "============================================================"
echo " All done!  Figures are in results/figures/"
echo "============================================================"
ls -lh results/figures/
