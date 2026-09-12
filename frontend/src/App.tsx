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
  const [controls, setControls] = useState<ControlsState>(DEFAULT_CONTROLS);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selected, setSelected] = useState<SelectedVoxel | null>(null);
  const [fps, setFps] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [gpu, setGpu] = useState('CPU');
  const [manifest, setManifest] = useState<SceneManifest | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);

  const voxelsRef = useRef<VoxelData[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(performance.now());
  const sendTimeRef = useRef(performance.now());
  const latencySumRef = useRef(0);
  const pendingFrameRef = useRef<ArrayBuffer | null>(null);
  const rafRef = useRef<number>(0);
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

    const consumeFrame = (buffer: ArrayBuffer) => {
      const receiveTime = performance.now();
      const count = new Uint32Array(buffer, 0, 1)[0];
      const floats = new Float32Array(buffer, 4, count * 4);
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
      voxelsRef.current = voxelList;

      const interval = receiveTime - sendTimeRef.current;
      sendTimeRef.current = receiveTime;
      latencySumRef.current += interval;
      frameCountRef.current += 1;
      if (receiveTime - lastTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        setLatencyMs(latencySumRef.current / Math.max(frameCountRef.current, 1));
        frameCountRef.current = 0;
        latencySumRef.current = 0;
        lastTimeRef.current = receiveTime;
      }
    };

    ws.onmessage = (event: MessageEvent) => {
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
          voxelsRef={voxelsRef}
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
