import React from 'react';

interface ControlPanelProps {
  threshold: number;
  setThreshold: (val: number) => void;
  isPaused: boolean;
  setIsPaused: (val: boolean) => void;
  activeVoxelCount: number;
  fps: number;
  latencyMs: number;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  threshold,
  setThreshold,
  isPaused,
  setIsPaused,
  activeVoxelCount,
  fps,
  latencyMs,
}) => {
  return (
    <div
      style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        zIndex: 10,
        width: '320px',
        padding: '20px',
        background: 'rgba(22, 27, 34, 0.85)',
        backdropFilter: 'blur(10px)',
        borderRadius: '12px',
        border: '1px solid #30363d',
        color: '#c9d1d9',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
      }}
    >
      <h2 style={{ margin: '0 0 12px 0', fontSize: '18px', color: '#58a6ff' }}>
        VisionOccupancy 
      </h2>

      {/* User-Driven Interaction: Threshold Slider */}
      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px' }}>
          Occupancy Threshold: <strong>{threshold.toFixed(2)}</strong>
        </label>
        <input
          type="range"
          min="0.1"
          max="0.9"
          step="0.02"
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          style={{ width: '100%', cursor: 'pointer' }}
        />
      </div>

      {/* User-Driven Interaction: Stream Control */}
      <button
        onClick={() => setIsPaused(!isPaused)}
        style={{
          width: '100%',
          padding: '10px',
          borderRadius: '6px',
          border: 'none',
          background: isPaused ? '#238636' : '#da3633',
          color: '#ffffff',
          fontWeight: 'bold',
          cursor: 'pointer',
          marginBottom: '16px',
        }}
      >
        {isPaused ? '▶ Resume Stream' : '⏸ Pause Stream'}
      </button>

      {/* Telemetry HUD */}
      <div style={{ fontSize: '12px', borderTop: '1px solid #30363d', paddingTop: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
          <span>Active 3D Voxels:</span>
          <strong style={{ color: '#7ee787' }}>{activeVoxelCount.toLocaleString()}</strong>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
          <span>Client Render FPS:</span>
          <strong style={{ color: fps > 45 ? '#7ee787' : '#ffa657' }}>{fps} FPS</strong>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>WebSocket Latency:</span>
          <strong style={{ color: latencyMs < 30 ? '#7ee787' : '#f85149' }}>{latencyMs} ms</strong>
        </div>
      </div>
    </div>
  );
};