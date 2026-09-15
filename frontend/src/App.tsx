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
  if (z < 0.45) return 'driveable';
  if (z < 2.3) return 'vehicle';
  return 'pedestrian';
}

function depthSourceFor(voxel: VoxelData): { label: string; id: string } {
  // nuScenes ego: x forward, y left, z up
  const forward = voxel.x;
  const left = voxel.y;
  if (forward >= 4) {
    if (left > 4) return { label: 'Front-Left Cam', id: 'CAM_FRONT_LEFT' };
    if (left < -4) return { label: 'Front-Right Cam', id: 'CAM_FRONT_RIGHT' };
    return { label: 'Front Cam', id: 'CAM_FRONT' };
  }
  if (forward <= -2) {
    if (left > 4) return { label: 'Back-Left Cam', id: 'CAM_BACK_LEFT' };
    if (left < -4) return { label: 'Back-Right Cam', id: 'CAM_BACK_RIGHT' };
    return { label: 'Back Cam', id: 'CAM_BACK' };
  }
  return left >= 0
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
  const pendingFrameRef = useRef<ArrayBuffer | null>(null);
  const rafRef = useRef<number>(0);
  const thresholdRef = useRef(controls.threshold);
  const voxelSizeRef = useRef(controls.voxelSize);
  const frameIndexRef = useRef(frameIndex);
  thresholdRef.current = controls.threshold;
  voxelSizeRef.current = controls.voxelSize;
  frameIndexRef.current = frameIndex;

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
  }, []);

  useEffect(() => {
    const ws = new WebSocket(wsUrl('/ws/occupancy'));
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      sendOccupancyConfig(ws);
    };

    const consumeFrame = (buffer: ArrayBuffer) => {
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
          };
          if (msg.type === 'occupancy_meta') {
            const elapsed =
              msg.elapsed_ms ?? (msg.depth_ms ?? 0) + (msg.project_ms ?? 0);
            setLatencyMs(elapsed);
            if (msg.device) {
              const source = msg.voxel_source ? ` · ${msg.voxel_source}` : '';
              setGpu(`${msg.device === 'cpu' ? 'CPU' : msg.device}${source}`);
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

  const handleSelectVoxel = useCallback((voxel: VoxelData | null) => {
    if (!voxel) {
      setSelected(null);
      return;
    }
    const source = depthSourceFor(voxel);
    const t = performance.now() / 1000;
    setSelected({
      voxel,
      meters: { x: voxel.x, y: voxel.y, z: voxel.z },
      depthSource: source.label,
      flow: {
        vx: Math.sin(t + voxel.x * 0.2) * 4.5,
        vy: Math.cos(t + voxel.y * 0.2) * 1.2,
      },
    });
  }, []);

  const highlightCamera = selected ? depthSourceFor(selected.voxel).id : null;

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
