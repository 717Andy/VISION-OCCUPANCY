# VisionOccupancy

React workbench plus a FastAPI occupancy engine. Run the two processes from separate terminals.

## Backend

The server reads `backend/requirements.txt`. `PIL` comes from **Pillow**; `open3d` is optional.

```bash
cd backend
conda deactivate                    # if the prompt still shows (base) next to (.venv)
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Use `python -m pip` and `python -m uvicorn` so packages go into this venv, not Anaconda `base`. In VS Code / Cursor, pick `backend/.venv` as the Python interpreter (status bar) so import warnings match that environment.

Confirm:

```bash
python -c "from PIL import Image; import numpy, open3d; print('ok')"
curl -s http://127.0.0.1:8000/api/health
```

If `open3d` is missing, the API still starts and occupancy uses a NumPy grid. If `PIL` is missing, install requirements again in the active venv.

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` and `/ws` to port 8000.
