import React from 'react';
import type { ViewMode } from './types';

interface TopBarProps {
  fps: number;
  latencyMs: number;
  miou: string;
  gpu: string;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  fps,
  latencyMs,
  miou,
  gpu,
  viewMode,
  onViewModeChange,
}) => {
  return (
    <header className="chrome-bar">
      <div className="brand">VisionOccupancy</div>
      <div className="mode-toggle" role="radiogroup" aria-label="Viewport mode">
        <span>Mode: [</span>
        <button
          type="button"
          role="radio"
          aria-checked={viewMode === 'single'}
          className={viewMode === 'single' ? 'active' : undefined}
          onClick={() => onViewModeChange('single')}
        >
          Single
        </button>
        <span>|</span>
        <button
          type="button"
          role="radio"
          aria-checked={viewMode === 'split'}
          className={viewMode === 'split' ? 'active' : undefined}
          onClick={() => onViewModeChange('split')}
        >
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
