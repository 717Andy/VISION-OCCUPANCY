import React from 'react';
import type { ViewMode } from './types';

interface TopBarProps {
  fps: number;
  latencyMs: number;
  liveIou: string;
  protocolIou: string;
  gpu: string;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  scenes: { name: string; label: string }[];
  scene: string;
  onSceneChange: (scene: string) => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  fps,
  latencyMs,
  liveIou,
  protocolIou,
  gpu,
  viewMode,
  onViewModeChange,
  scenes,
  scene,
  onSceneChange,
}) => {
  return (
    <header className="chrome-bar">
      <div className="brand">VisionOccupancy</div>
      <label className="scene-picker">
        Scene
        <select aria-label="nuScenes scene" value={scene} onChange={(event) => onSceneChange(event.target.value)}>
          {scenes.map((item) => (
            <option key={item.name} value={item.name}>
              {item.label}
            </option>
          ))}
        </select>
      </label>
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
          Split
        </button>
        <span>]</span>
      </div>
      <div className="hud">
        <span className="hud-text">
          HUD: {fps} FPS | Latency: {latencyMs.toFixed(1)}ms | Live {liveIou} · Protocol {protocolIou} | GPU: {gpu}
        </span>
      </div>
    </header>
  );
};
