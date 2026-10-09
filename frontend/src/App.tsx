import React, { useCallback, useEffect, useRef, useState } from 'react';
import { VoxelCanvas, createSharedOrbit } from './VoxelCanvas';
import { ControlPanel } from './ControlPanel';
import { CameraFeed } from './CameraFeed';
import { PlaybackBar } from './PlaybackBar';
import { TopBar } from './TopBar';
import { InspectionPanel } from './InspectionPanel';
import { CornerDock } from './CornerDock';
import { fetchGpuLabel, fetchManifest, fetchModels, fetchRun, fetchRuns, wsUrl } from './api';
import type {
  BenchmarkTable as BenchmarkTableData,
  ControlsState,
  SavedRun,
  SavedRunSummary,
  SceneManifest,
  SelectedVoxel,
  SemanticClass,
  DistanceZone,
  MiouDrop,
  ViewMode,
  VoxelData,
} from './types';

function classifyVoxel(x: number, y: number, z: number, _prob: number): SemanticClass {
  if (z < 0.45) return 'driveable';
  if (z < 2.3) return 'vehicle';
  return 'pedestrian';
}

const CAMERA_LABELS: Record<string, string> = {
  CAM_FRONT: 'Front Cam',
  CAM_FRONT_LEFT: 'Front-Left Cam',
  CAM_FRONT_RIGHT: 'Front-Right Cam',
  CAM_BACK: 'Back Cam',
  CAM_BACK_LEFT: 'Back-Left Cam',
  CAM_BACK_RIGHT: 'Back-Right Cam',
};

function rotationFromQuat(rotation: number[]): number[][] {
  const [w, x, y, z] = rotation;
  const n = w * w + x * x + y * y + z * z;
  const s = n < 1e-12 ? 0 : 2 / n;
  const wx = s * w * x;
  const wy = s * w * y;
  const wz = s * w * z;
  const xx = s * x * x;
  const xy = s * x * y;
  const xz = s * x * z;
  const yy = s * y * y;
  const yz = s * y * z;
  const zz = s * z * z;
  return [
    [1 - (yy + zz), xy - wz, xz + wy],
    [xy + wz, 1 - (xx + zz), yz - wx],
    [xz - wy, yz + wx, 1 - (xx + yy)],
  ];
}

function camerasSeeingPoint(
  voxel: VoxelData,
  calibration: Record<string, { intrinsic: number[][]; translation: number[]; rotation: number[] }> | undefined,
): { id: string; label: string; depth: number }[] {
  if (!calibration) return [];
  const seen: { id: string; label: string; depth: number }[] = [];
  for (const [id, cal] of Object.entries(calibration)) {
    const rotation = rotationFromQuat(cal.rotation);
    const dx = voxel.x - cal.translation[0];
    const dy = voxel.y - cal.translation[1];
    const dz = voxel.z - cal.translation[2];
    const cx = rotation[0][0] * dx + rotation[1][0] * dy + rotation[2][0] * dz;
    const cy = rotation[0][1] * dx + rotation[1][1] * dy + rotation[2][1] * dz;
    const cz = rotation[0][2] * dx + rotation[1][2] * dy + rotation[2][2] * dz;
    if (cz <= 0.5) continue;
    const u = (cal.intrinsic[0][0] * cx + cal.intrinsic[0][1] * cy + cal.intrinsic[0][2] * cz) / cz;
    const v = (cal.intrinsic[1][0] * cx + cal.intrinsic[1][1] * cy + cal.intrinsic[1][2] * cz) / cz;
    if (u >= 0 && v >= 0 && u < 1600 && v < 900) {
      seen.push({ id, label: CAMERA_LABELS[id] ?? id, depth: cz });
    }
  }
  seen.sort((a, b) => a.depth - b.depth);
  return seen;
}

function heightBandLabel(z: number): string {
  if (z < 0.45) return 'Below 0.45 m';
  if (z < 2.3) return '0.45 to 2.3 m';
  return 'Above 2.3 m';
}

const DEFAULT_CONTROLS: ControlsState = {
  voxelSize: 0.2,
  threshold: 0.38,
  layers: { driveable: true, vehicle: true, pedestrian: true },
};

export const App: React.FC = () => {
  const [controls, setControls] = useState<ControlsState>(DEFAULT_CONTROLS);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selected, setSelected] = useState<SelectedVoxel | null>(null);
  const [fps, setFps] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [gpu, setGpu] = useState('CPU');
  const [manifest, setManifest] = useState<SceneManifest | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('single');
  const [miou, setMiou] = useState('—');
  const [distanceZones, setDistanceZones] = useState<DistanceZone[]>([]);
  const [miouDrop, setMiouDrop] = useState<MiouDrop>({});
  const [discrepancyOn, setDiscrepancyOn] = useState(false);
  const [discrepancyColor, setDiscrepancyColor] = useState('#f0883e');
  const [benchmark, setBenchmark] = useState<BenchmarkTableData | null>(null);
  const [runs, setRuns] = useState<SavedRunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<SavedRun | null>(null);
  const [freeOn, setFreeOn] = useState(false);
  const [unknownOn, setUnknownOn] = useState(false);
  const [predictionModel, setPredictionModel] = useState('Monocular depth');
  const [modelNames, setModelNames] = useState<string[]>([
    'Monocular depth',
    'VoxNet 3D CNN',
    'Ground-plane IPM',
    'Multi-view stereo',
    'Lift-Splat',
  ]);

  const predVoxelsRef = useRef<VoxelData[]>([]);
  const gtVoxelsRef = useRef<VoxelData[]>([]);
  const errorVoxelsRef = useRef<VoxelData[]>([]);
  const freeVoxelsRef = useRef<VoxelData[]>([]);
  const unknownVoxelsRef = useRef<VoxelData[]>([]);
  const orbitRef = useRef(createSharedOrbit());
  const wsRef = useRef<WebSocket | null>(null);
  const pendingFrameRef = useRef<ArrayBuffer | null>(null);
  const rafRef = useRef<number>(0);
  const thresholdRef = useRef(controls.threshold);
  const voxelSizeRef = useRef(controls.voxelSize);
  const frameIndexRef = useRef(frameIndex);
  const modelRef = useRef(predictionModel);
  thresholdRef.current = controls.threshold;
  voxelSizeRef.current = controls.voxelSize;
  frameIndexRef.current = frameIndex;
  modelRef.current = predictionModel;

  const sendOccupancyConfig = useCallback((socket?: WebSocket | null) => {
    const ws = socket ?? wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(
      JSON.stringify({
        type: 'config',
        frame_index: frameIndexRef.current,
        voxel_size: voxelSizeRef.current,
        occupancy_threshold: thresholdRef.current,
        threshold: thresholdRef.current,
        model: modelRef.current,
        paused: false,
      }),
    );
  }, []);

  useEffect(() => {
    fetchManifest()
      .then((data) => {
        setManifest(data);
        setFrameIndex(0);
      })
      .catch((err) => console.warn('nuScenes manifest unavailable', err));
    fetchGpuLabel().then(setGpu);
    fetchRuns().then(setRuns);
    fetchModels().then((names) => {
      if (names.length > 0) setModelNames(names);
    });
  }, []);

  useEffect(() => {
    sendOccupancyConfig();
  }, [predictionModel, sendOccupancyConfig]);

  useEffect(() => {
    const ws = new WebSocket(wsUrl('/ws/occupancy'));
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      sendOccupancyConfig(ws);
    };

    const decodeVoxels = (floats: Float32Array, count: number): VoxelData[] => {
      const stride = Math.max(1, Math.ceil(count / 900));
      const kept = Math.ceil(count / stride);
      const voxelList: VoxelData[] = new Array(kept);
      let written = 0;
      for (let i = 0; i < count; i += stride) {
        const base = i * 4;
        const x = floats[base];
        const y = floats[base + 1];
        const z = floats[base + 2];
        const prob = floats[base + 3];
        voxelList[written] = { x, y, z, prob, cls: classifyVoxel(x, y, z, prob) };
        written += 1;
      }
      return voxelList;
    };

    const consumeFrame = (buffer: ArrayBuffer) => {
      if (buffer.byteLength < 20) return;
      const header = new Uint32Array(buffer, 0, 5);
      const predCount = header[0];
      const gtCount = header[1];
      const errCount = header[2];
      const freeCount = header[3];
      const unknownCount = header[4];
      const predBytes = predCount * 16;
      const gtBytes = gtCount * 16;
      const errBytes = errCount * 16;
      const freeBytes = freeCount * 16;
      const unknownBytes = unknownCount * 16;
      if (buffer.byteLength < 20 + predBytes + gtBytes + errBytes + freeBytes + unknownBytes) return;
      let offset = 20;
      const predFloats = new Float32Array(buffer, offset, predCount * 4);
      offset += predBytes;
      const gtFloats = new Float32Array(buffer, offset, gtCount * 4);
      offset += gtBytes;
      const errFloats = new Float32Array(buffer, offset, errCount * 4);
      offset += errBytes;
      const freeFloats = new Float32Array(buffer, offset, freeCount * 4);
      offset += freeBytes;
      const unknownFloats = new Float32Array(buffer, offset, unknownCount * 4);
      predVoxelsRef.current = decodeVoxels(predFloats, predCount);
      gtVoxelsRef.current = decodeVoxels(gtFloats, gtCount);
      errorVoxelsRef.current = decodeVoxels(errFloats, errCount);
      freeVoxelsRef.current = decodeVoxels(freeFloats, freeCount);
      unknownVoxelsRef.current = decodeVoxels(unknownFloats, unknownCount);
    };

    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data) as {
            type?: string;
            elapsed_ms?: number;
            depth_ms?: number;
            project_ms?: number;
            device?: string;
            voxel_source?: string;
            miou?: number;
            metric?: {
              grid_m?: number;
              tolerance_cells?: number;
              unknown?: string;
              iou?: number;
              distance_zones?: DistanceZone[];
              miou_drop?: MiouDrop;
            };
            benchmark?: BenchmarkTableData | null;
          };
          if (msg.type === 'occupancy_meta') {
            const elapsed =
              msg.elapsed_ms ?? (msg.depth_ms ?? 0) + (msg.project_ms ?? 0);
            setLatencyMs(elapsed);
            if (msg.device) {
              const source = msg.voxel_source ? ` · ${msg.voxel_source}` : '';
              setGpu(`${msg.device === 'cpu' ? 'CPU' : msg.device}${source}`);
            }
            if (msg.metric && typeof msg.metric.iou === 'number') {
              const grid = msg.metric.grid_m ?? 0.2;
              const tol = msg.metric.tolerance_cells ?? 1;
              const unknown = msg.metric.unknown === 'ignored' ? 'unknown ignored' : 'unknown as free';
              setMiou(`${msg.metric.iou.toFixed(2)} (${grid} m, tol ${tol}, ${unknown})`);
              setDistanceZones(Array.isArray(msg.metric.distance_zones) ? msg.metric.distance_zones : []);
              setMiouDrop(msg.metric.miou_drop ?? {});
            } else if (typeof msg.miou === 'number' && Number.isFinite(msg.miou)) {
              setMiou(msg.miou.toFixed(2));
            }
            if ('benchmark' in msg) {
              setBenchmark(msg.benchmark ?? null);
            }
          }
        } catch {
          /* ignore malformed control frames */
        }
        return;
      }
      if (!(event.data instanceof ArrayBuffer)) return;
      pendingFrameRef.current = event.data;
      if (rafRef.current) return;
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = 0;
        const pending = pendingFrameRef.current;
        pendingFrameRef.current = null;
        if (pending) consumeFrame(pending);
      });
    };

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      ws.close();
    };
  }, [sendOccupancyConfig]);

  useEffect(() => {
    let frames = 0;
    let last = performance.now();
    let id = 0;
    const loop = (now: number) => {
      frames += 1;
      if (now - last >= 1000) {
        setFps(frames);
        frames = 0;
        last = now;
      }
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    sendOccupancyConfig();
  }, [frameIndex, controls.voxelSize, controls.threshold, sendOccupancyConfig]);

  useEffect(() => {
    if (!playing || !manifest) return;
    const intervalMs = 1000 / Math.max(manifest.sample_hz, 1);
    const id = window.setInterval(() => {
      setFrameIndex((prev) => {
        const last = Math.max(manifest.frame_count - 1, 0);
        return prev >= last ? 0 : prev + 1;
      });
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [playing, manifest]);

  const handleSelectVoxel = useCallback((voxel: VoxelData | null, kind: 'lidar' | 'prediction' = 'prediction') => {
    if (!voxel) {
      setSelected(null);
      return;
    }
    const calibration = manifest?.frames?.[frameIndexRef.current]?.calibration;
    const seen = camerasSeeingPoint(voxel, calibration);
    setSelected({
      voxel,
      meters: { x: voxel.x, y: voxel.y, z: voxel.z },
      source: kind === 'lidar' ? 'Lidar return' : 'Predicted occupancy',
      cameras: seen.length
        ? seen.map((camera) => camera.label).join(', ')
        : 'None of the six cameras see this cell',
    });
  }, [manifest]);

  const handleSelectRun = useCallback((runId: string) => {
    fetchRun(runId).then(setSelectedRun);
  }, []);

  const highlightCamera = selected
    ? camerasSeeingPoint(
        selected.voxel,
        manifest?.frames?.[frameIndex]?.calibration,
      )[0]?.id ?? null
    : null;

  useEffect(() => {
    if (viewMode === 'split') {
      orbitRef.current.driver = 'pred';
    }
  }, [viewMode]);

  return (
    <div className="app-shell">
      <TopBar
        fps={fps}
        latencyMs={latencyMs}
        miou={miou}
        gpu={gpu}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
      />

      <div className={`workspace${viewMode === 'split' ? ' split' : ''}`}>
        <CameraFeed
          manifest={manifest}
          frameIndex={frameIndex}
          settingsOpen={settingsOpen}
          onToggleSettings={() => {
            setSettingsOpen((open) => !open);
            setSelected(null);
          }}
          highlightCamera={highlightCamera}
        />
        {viewMode === 'split' && (
          <VoxelCanvas
            className="pane pane-voxel pane-gt"
            title="Ground Truth"
            subtitle="Camera-visible lidar, real hits"
            voxelsRef={gtVoxelsRef}
            voxelSize={controls.voxelSize}
            layers={controls.layers}
            selected={selected}
            onSelect={(voxel) => handleSelectVoxel(voxel, 'lidar')}
            orbitRef={orbitRef}
            orbitId="gt"
          />
        )}
        <VoxelCanvas
          className="pane pane-voxel pane-pred"
          title="Vision Prediction"
          subtitle={predictionModel}
          models={modelNames}
          activeModel={predictionModel}
          onModelChange={setPredictionModel}
          voxelsRef={predVoxelsRef}
          voxelSize={controls.voxelSize}
          layers={controls.layers}
          selected={selected}
          onSelect={(voxel) => handleSelectVoxel(voxel, 'prediction')}
          orbitRef={orbitRef}
          orbitId="pred"
          discrepancyRef={errorVoxelsRef}
          discrepancyEnabled={discrepancyOn}
          discrepancyColor={discrepancyColor}
          onDiscrepancyEnabledChange={setDiscrepancyOn}
          onDiscrepancyColorChange={setDiscrepancyColor}
          freeRef={freeVoxelsRef}
          freeEnabled={freeOn}
          onFreeEnabledChange={setFreeOn}
          unknownRef={unknownVoxelsRef}
          unknownEnabled={unknownOn}
          onUnknownEnabledChange={setUnknownOn}
        />
        {settingsOpen && (
          <ControlPanel
            controls={controls}
            onChange={setControls}
            onClose={() => setSettingsOpen(false)}
          />
        )}
        {selected && !settingsOpen && (
          <InspectionPanel selected={selected} onClose={() => setSelected(null)} />
        )}
        <CornerDock
          table={benchmark}
          runs={runs.filter((run) => !manifest?.scene_name || run.scene_name === manifest.scene_name)}
          selectedRun={selectedRun}
          onSelectRun={handleSelectRun}
          distanceZones={distanceZones}
          miouDrop={miouDrop}
        />
      </div>

      <PlaybackBar
        playing={playing}
        onTogglePlay={() => setPlaying((p) => !p)}
        frameIndex={frameIndex}
        frameCount={manifest?.frame_count ?? 1}
        durationS={manifest?.duration_s ?? 20}
        sampleHz={manifest?.sample_hz ?? 2}
        onSeek={(frame) => {
          setPlaying(false);
          setFrameIndex(frame);
        }}
      />
    </div>
  );
};

export default App;
