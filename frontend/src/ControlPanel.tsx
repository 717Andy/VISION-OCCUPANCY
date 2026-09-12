import React from 'react';
import type { ControlsState, SemanticClass } from './types';

interface ControlPanelProps {
  controls: ControlsState;
  onChange: (next: ControlsState) => void;
  onClose: () => void;
}

const LAYERS: { id: SemanticClass; label: string }[] = [
  { id: 'driveable', label: 'Driveable Area' },
  { id: 'vehicle', label: 'Vehicle' },
  { id: 'pedestrian', label: 'Pedestrians' },
];

export const ControlPanel: React.FC<ControlPanelProps> = ({ controls, onChange, onClose }) => {
  const voxelFill = ((controls.voxelSize - 0.1) / 0.4) * 100;
  const threshFill = (controls.threshold / 1) * 100;

  return (
    <aside className="overlay-card controls-card" role="dialog" aria-label="Controls">
      <div className="card-head">
        <h2>Controls</h2>
        <button type="button" className="close-btn" onClick={onClose} aria-label="Close controls">
          X
        </button>
      </div>

      <div className="control-block">
        <label htmlFor="voxel-size">Voxel Size:</label>
        <div className="slider-row">
          <input
            id="voxel-size"
            type="range"
            min={0.1}
            max={0.5}
            step={0.05}
            value={controls.voxelSize}
            style={{ ['--fill' as string]: `${voxelFill}%` }}
            onChange={(e) => onChange({ ...controls, voxelSize: Number(e.target.value) })}
          />
          <span className="value">{controls.voxelSize.toFixed(1)}m</span>
        </div>
      </div>

      <div className="control-block">
        <label htmlFor="occupancy-threshold">Occupancy Threshold:</label>
        <div className="slider-row">
          <input
            id="occupancy-threshold"
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={controls.threshold}
            style={{ ['--fill' as string]: `${threshFill}%` }}
            onChange={(e) => onChange({ ...controls, threshold: Number(e.target.value) })}
          />
          <span className="value">{controls.threshold.toFixed(2)}</span>
        </div>
      </div>

      <div className="control-block">
        <label>Semantic Layers:</label>
        <div className="layer-list">
          {LAYERS.map((layer) => (
            <label key={layer.id} className="layer-item">
              <input
                type="checkbox"
                checked={controls.layers[layer.id]}
                onChange={(e) =>
                  onChange({
                    ...controls,
                    layers: { ...controls.layers, [layer.id]: e.target.checked },
                  })
                }
              />
              {layer.label}
            </label>
          ))}
        </div>
      </div>
    </aside>
  );
};
