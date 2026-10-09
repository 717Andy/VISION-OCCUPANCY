import React from 'react';
import type { ControlsState } from './types';
import { HEIGHT_BANDS } from './heightBands';
import { useCardDrag } from './useCardDrag';

interface ControlPanelProps {
  controls: ControlsState;
  onChange: (next: ControlsState) => void;
  onClose: () => void;
  discrepancyOn: boolean;
  discrepancyColor: string;
  onDiscrepancyEnabledChange: (enabled: boolean) => void;
  onDiscrepancyColorChange: (color: string) => void;
  freeOn: boolean;
  onFreeEnabledChange: (enabled: boolean) => void;
  unknownOn: boolean;
  onUnknownEnabledChange: (enabled: boolean) => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  controls,
  onChange,
  onClose,
  discrepancyOn,
  discrepancyColor,
  onDiscrepancyEnabledChange,
  onDiscrepancyColorChange,
  freeOn,
  onFreeEnabledChange,
  unknownOn,
  onUnknownEnabledChange,
}) => {
  const voxelFill = ((controls.voxelSize - 0.1) / 0.4) * 100;
  const threshFill = (controls.threshold / 1) * 100;
  const { dragStyle, onPointerDown } = useCardDrag();

  return (
    <aside className="overlay-card controls-card" role="dialog" aria-label="Controls" style={dragStyle}>
      <div className="card-head" onPointerDown={onPointerDown}>
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
        <label>Height:</label>
        <div className="layer-list">
          {HEIGHT_BANDS.map((layer) => (
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
              <i className="height-swatch" style={{ background: layer.color }} />
              {layer.label}
            </label>
          ))}
        </div>
      </div>

      <div className="control-block">
        <div className="view-toggles">
          <button
            type="button"
            className="discrepancy-btn"
            aria-pressed={discrepancyOn}
            style={discrepancyOn ? { borderColor: discrepancyColor } : undefined}
            onClick={() => onDiscrepancyEnabledChange(!discrepancyOn)}
          >
            Discrepancy
          </button>
          <input
            type="color"
            className="discrepancy-color"
            aria-label="Discrepancy color"
            value={discrepancyColor}
            onChange={(event) => onDiscrepancyColorChange(event.target.value)}
          />
          <button
            type="button"
            className="discrepancy-btn"
            aria-pressed={freeOn}
            onClick={() => onFreeEnabledChange(!freeOn)}
          >
            Free
          </button>
          <button
            type="button"
            className="discrepancy-btn"
            aria-pressed={unknownOn}
            onClick={() => onUnknownEnabledChange(!unknownOn)}
          >
            Unknown
          </button>
        </div>
      </div>
    </aside>
  );
};
