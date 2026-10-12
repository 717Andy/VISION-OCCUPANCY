#!/usr/bin/env bash
# Start the occupancy API and the UI. Ctrl-C stops both.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT/backend"
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$ROOT/frontend"
npm run dev -- --host 127.0.0.1 --port 5173
