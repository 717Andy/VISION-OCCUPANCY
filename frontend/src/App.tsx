import React, { useCallback, useEffect, useRef, useState } from 'react';
import { VoxelCanvas } from './VoxelCanvas';
import { ControlPanel } from './ControlPanel';
import { CameraFeed } from './CameraFeed';
import { PlaybackBar } from './PlaybackBar';
import { TopBar } from './TopBar';
import { InspectionPanel } from './InspectionPanel';
import { fetchGpuLabel, fetchManifest, wsUrl } from './api';
import type {
  ControlsState,
  SceneManifest,
  SelectedVoxel,
  SemanticClass,
  VoxelData,
} from './types';

function classifyVoxel(x: number, y: number, z: number, _prob: number): SemanticClass {
  if (z < 3) return 'driveable';
  if (z < 8 && Math.abs(x) < 8) return 'vehicle';
  return 'pedestrian';
}

function depthSourceFor(voxel: VoxelData): { label: string; id: string } {
  const ax = voxel.x;
  const ay = voxel.y;
  if (Math.abs(ay) >= Math.abs(ax)) {
    if (ay >= 0) {
      if (ax < -4) return { label: 'Front-Left Cam', id: 'CAM_FRONT_LEFT' };
      if (ax > 4) return { label: 'Front-Right Cam', id: 'CAM_FRONT_RIGHT' };
      return { label: 'Front Cam', id: 'CAM_FRONT' };
    }
    if (ax < -4) return { label: 'Back-Left Cam', id: 'CAM_BACK_LEFT' };
    if (ax > 4) return { label: 'Back-Right Cam', id: 'CAM_BACK_RIGHT' };
    return { label: 'Back Cam', id: 'CAM_BACK' };
  }
  return ax < 0
    ? { label: 'Front-Left Cam', id: 'CAM_FRONT_LEFT' }
    : { label: 'Front-Right Cam', id: 'CAM_FRONT_RIGHT' };
}

const DEFAULT_CONTROLS: ControlsState = {
  voxelSize: 0.2,
  threshold: 0.38,
  layers: { driveable: true, vehicle: true, pedestrian: true },
};

export const App: React.FC = () => {
  const [voxels, setVoxels] = useState<VoxelData[]>([]);
  const [controls, setControls] = useState<ControlsState>(DEFAULT_CONTROLS);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selected, setSelected] = useState<SelectedVoxel | null>(null);
  const [fps, setFps] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [gpu, setGpu] = useState('CPU');
  const [manifest, setManifest] = useState<SceneManifest | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(performance.now());
  const sendTimeRef = useRef(performance.now());
  const thresholdRef = useRef(controls.threshold);
  thresholdRef.current = controls.threshold;

  useEffect(() => {
    fetchManifest()
      .then((data) => {
        setManifest(data);
        setFrameIndex(0);
      })
      .catch((err) => console.warn('nuScenes manifest unavailable', err));
    fetchGpuLabel().then(setGpu);
  }, []);

  useEffect(() => {
    const ws = new WebSocket(wsUrl('/ws/occupancy'));
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ threshold: thresholdRef.current, paused: false }));
    };

    ws.onmessage = (event: MessageEvent) => {
      const receiveTime = performance.now();
      setLatencyMs(receiveTime - sendTimeRef.current);
      sendTimeRef.current = receiveTime;

      if (!(event.data instanceof ArrayBuffer)) return;
      const view = new DataView(event.data);
      const count = view.getUint32(0, true);
      const voxelList: VoxelData[] = [];
      let offset = 4;
      for (let i = 0; i < count; i++) {
        const x = view.getFloat32(offset, true);
        const y = view.getFloat32(offset + 4, true);
        const z = view.getFloat32(offset + 8, true);
        const prob = view.getFloat32(offset + 12, true);
        offset += 16;
        voxelList.push({ x, y, z, prob, cls: classifyVoxel(x, y, z, prob) });
      }
      setVoxels(voxelList);

      frameCountRef.current += 1;
      const now = performance.now();
      if (now - lastTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastTimeRef.current = now;
      }
    };

    return () => ws.close();
  }, []);

  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ threshold: controls.threshold, paused: false }));
    }
  }, [controls.threshold]);

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

  const handleSelectVoxel = useCallback(
    (voxel: VoxelData | null) => {
      if (!voxel) {
        setSelected(null);
        return;
      }
      const source = depthSourceFor(voxel);
      const t = performance.now() / 1000;
      setSelected({
        voxel,
        meters: {
          x: voxel.x * controls.voxelSize,
          y: voxel.y * controls.voxelSize,
          z: voxel.z * controls.voxelSize,
        },
        depthSource: source.label,
        flow: {
          vx: Math.sin(t + voxel.x * 0.2) * 4.5,
          vy: Math.cos(t + voxel.y * 0.2) * 1.2,
        },
      });
    },
    [controls.voxelSize],
  );

  const highlightCamera = selected
    ? depthSourceFor(selected.voxel).id
    : null;

  return (
    <div className="app-shell">
      <TopBar fps={fps} latencyMs={latencyMs} miou="—" gpu={gpu} />

      <div className="workspace">
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
        <VoxelCanvas
          voxels={voxels}
          voxelSize={controls.voxelSize}
          layers={controls.layers}
          selected={selected}
          onSelect={handleSelectVoxel}
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
