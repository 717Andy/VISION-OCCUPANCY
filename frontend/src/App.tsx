import React, { useEffect, useState, useRef } from 'react';
import { VoxelCanvas } from './VoxelCanvas';
import { ControlPanel } from './ControlPanel';

interface VoxelData {
  x: number;
  y: number;
  z: number;
  prob: number;
}

export const App: React.FC = () => {
  const [voxels, setVoxels] = useState<VoxelData[]>([]);
  const [threshold, setThreshold] = useState<number>(0.4);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);

  const wsRef = useRef<WebSocket | null>(null);
  const frameCountRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(performance.now());
  const sendTimeRef = useRef<number>(performance.now());

  // Establish WebSocket connection & Binary Parser
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/occupancy');
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => console.log('Connected to Backend Pipeline.');

    ws.onmessage = (event: MessageEvent) => {
      const receiveTime = performance.now();
      setLatencyMs(Math.round(receiveTime - sendTimeRef.current));
      sendTimeRef.current = receiveTime;

      if (event.data instanceof ArrayBuffer) {
        const buffer = event.data;
        const view = new DataView(buffer);
        
        // Read header uint32: total active voxels
        const count = view.getUint32(0, true);
        const voxelList: VoxelData[] = [];
        
        // Parse float32 array
        let offset = 4;
        for (let i = 0; i < count; i++) {
          const x = view.getFloat32(offset, true);
          const y = view.getFloat32(offset + 4, true);
          const z = view.getFloat32(offset + 8, true);
          const prob = view.getFloat32(offset + 12, true);
          offset += 16;
          
          voxelList.push({ x, y, z, prob });
        }
        
        setVoxels(voxelList);

        // Compute Client FPS
        frameCountRef.current += 1;
        const now = performance.now();
        if (now - lastTimeRef.current >= 1000) {
          setFps(frameCountRef.current);
          frameCountRef.current = 0;
          lastTimeRef.current = now;
        }
      }
    };

    return () => ws.close();
  }, []);

  // Send interactive slider threshold to Python backend
  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ threshold, paused: isPaused }));
    }
  }, [threshold, isPaused]);

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden' }}>
      <ControlPanel
        threshold={threshold}
        setThreshold={setThreshold}
        isPaused={isPaused}
        setIsPaused={setIsPaused}
        activeVoxelCount={voxels.length}
        fps={fps}
        latencyMs={latencyMs}
      />
      <VoxelCanvas voxels={voxels} />
    </div>
  );
};

export default App;