#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for VISION-OCCUPANCY.
# Prepares the FastAPI backend (Python venv) and the Vite/React frontend (npm).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# System dependency: the default image ships python3.12 without the venv module.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.12-venv
fi

# Backend: create/refresh the virtual environment and install pinned deps.
cd "$REPO_ROOT/backend"
if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Frontend: install pinned deps from the lockfile.
cd "$REPO_ROOT/frontend"
npm ci
