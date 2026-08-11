#!/usr/bin/env bash
# ============================================================
# start_dashboard.sh
# Starts the FastAPI data API + the React/TypeScript Vite UI.
# Usage: bash start_dashboard.sh
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "⚡ SmartGrid FL Dashboard Launcher"
echo "==================================="

# ── 1. Activate Python venv ──────────────────────────────────
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
  echo "✅ Python venv activated"
else
  echo "⚠  No venv found at venv/. Install requirements first."
fi

# ── 2. Start FastAPI backend ──────────────────────────────────
echo "🚀 Starting FastAPI backend on http://localhost:8008 ..."
python3 -m uvicorn dashboard.api:app --host 0.0.0.0 --port 8008 --reload &
API_PID=$!
echo "   API PID: $API_PID"
sleep 2

# ── 3. Start Vite dev server ──────────────────────────────────
echo "🌐 Starting React/TypeScript UI on http://localhost:5173 ..."
cd dashboard/ui
if [ ! -d "node_modules" ]; then
  echo "📦 Installing npm dependencies (first run) ..."
  npm install
fi
npm run dev &
UI_PID=$!
echo "   UI PID:  $UI_PID"
cd "$SCRIPT_DIR"

echo ""
echo "✅ Dashboard is live!"
echo "   → UI:  http://localhost:5173"
echo "   → API: http://localhost:8008/docs"
echo ""
echo "   Press Ctrl+C to stop both servers."
echo ""

# Wait and clean up on exit
trap "echo ''; echo 'Stopping…'; kill $API_PID $UI_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
