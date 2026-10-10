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
  ghostOn: boolean;
  onGhostEnabledChange: (enabled: boolean) => void;
  falsePositiveOn: boolean;
  onFalsePositiveEnabledChange: (enabled: boolean) => void;
  missOn: boolean;
  onMissEnabledChange: (enabled: boolean) => void;
}

const FALSE_POSITIVE_COLOR = '#ff5d73';
const MISS_COLOR = '#e6d35a';

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
  ghostOn,
  onGhostEnabledChange,
  falsePositiveOn,
  onFalsePositiveEnabledChange,
  missOn,
  onMissEnabledChange,
}) => {
  const voxelFill = ((controls.voxelSize - 0.1) / 0.4) * 100;
  const threshFill = (controls.threshold / 1) * 100;
  const opacityFill = ((controls.opacity - 0.15) / 0.85) * 100;
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
        <p className="control-note">Live view and live IoU. The protocol table stays on 1.0 m cells.</p>
      </div>

      <div className="control-block">
        <label htmlFor="occupancy-opacity">Opacity:</label>
        <div className="slider-row">
          <input
            id="occupancy-opacity"
            type="range"
            min={0.15}
            max={1}
            step={0.05}
            value={controls.opacity}
            style={{ ['--fill' as string]: `${opacityFill}%` }}
            onChange={(e) => onChange({ ...controls, opacity: Number(e.target.value) })}
          />
          <span className="value">{controls.opacity.toFixed(2)}</span>
        </div>
        <p className="control-note">Fades the occupancy cubes. Protocol false positives and misses stay solid.</p>
      </div>

      <div className="control-block">
        <label htmlFor="occupancy-threshold">Occupancy Threshold (MiDaS):</label>
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
          <button
            type="button"
            className="discrepancy-btn"
            aria-pressed={ghostOn}
            aria-label="Show lidar ground truth as a ghost overlay"
            style={ghostOn ? { borderColor: '#c5d4e0' } : undefined}
            onClick={() => onGhostEnabledChange(!ghostOn)}
          >
            GT Ghost
          </button>
          <button
            type="button"
            className="discrepancy-btn"
            aria-pressed={falsePositiveOn}
            aria-label="Show 1.0 m false positives"
            style={falsePositiveOn ? { borderColor: FALSE_POSITIVE_COLOR } : undefined}
            onClick={() => onFalsePositiveEnabledChange(!falsePositiveOn)}
          >
            False positive
          </button>
          <button
            type="button"
            className="discrepancy-btn"
            aria-pressed={missOn}
            aria-label="Show 1.0 m misses"
            style={missOn ? { borderColor: MISS_COLOR } : undefined}
            onClick={() => onMissEnabledChange(!missOn)}
          >
            Miss
          </button>
        </div>
      </div>
    </aside>
  );
};
