# VisionOccupancy

A camera occupancy workbench. Six nuScenes cameras, lidar ground truth, and several predictors, scored on one protocol.

## Protocol

Cells are 1.0 m. A prediction is scored only where the cameras can mark a cell occupied or free. Unknown cells are ignored.

IoU = TP / (TP + FP + FN).

scene-0103 (boston-seaport) is the training clip. Its score uses the held-out frames, every fifth frame. scene-0061 (onenorth), scene-0796 (queenstown), and scene-1094 (hollandvillage, night) are unseen. Every frame of those clips is scored, and the models were not trained on them.

The HUD "Live" number is the on-screen voxel size, 0.2 m by default, with one-cell tolerance. The protocol table is the 1.0 m score. The distance chart uses the live score.

## Run

From the repo root, after the dependencies below are installed:

```bash
./start.sh
```

The API listens on http://127.0.0.1:8000 and the UI on http://127.0.0.1:5173.

```bash
python3 -m pip install -r backend/requirements.txt
npm install --prefix frontend
```

The prepared images, lidar, and MiDaS depth maps under `backend/data/nuscenes` are part of this repo so the workbench opens without downloading nuScenes again.

## Tests

```bash
cd backend && python3 -m unittest discover -s . -p 'test_*.py'
```
