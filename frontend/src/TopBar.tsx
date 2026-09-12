import React from 'react';

interface TopBarProps {
  fps: number;
  latencyMs: number;
  miou: string;
  gpu: string;
}

export const TopBar: React.FC<TopBarProps> = ({ fps, latencyMs, miou, gpu }) => {
  return (
    <header className="chrome-bar">
      <div className="brand">VisionOccupancy</div>
      <div className="mode-toggle" aria-label="View mode">
        <span>Mode: [</span>
        <button type="button" className="active" aria-current="page">
          Single
        </button>
        <span>|</span>
        <button type="button" disabled title="Split Ground Truth view ships in a later iteration">
          Split GT
        </button>
        <span>]</span>
      </div>
      <div className="hud">
        HUD: {fps} FPS | Latency: {latencyMs.toFixed(1)}ms | mIoU: {miou} | GPU: {gpu}
      </div>
    </header>
  );
};
