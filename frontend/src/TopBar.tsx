import React from 'react';
import type { DistanceZone, ViewMode } from './types';

interface TopBarProps {
  fps: number;
  latencyMs: number;
  miou: string;
  gpu: string;
  distanceZones?: DistanceZone[];
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
}

function DistanceErrorChart({ zones }: { zones: DistanceZone[] }) {
  const width = 168;
  const height = 48;
  const left = 6;
  const right = 6;
  const top = 4;
  const bottom = 14;
  const innerWidth = width - left - right;
  const innerHeight = height - top - bottom;
  const points = zones.map((zone, index) => {
    if (typeof zone.error !== 'number' || !Number.isFinite(zone.error)) return null;
    const x = left + (index * innerWidth) / Math.max(zones.length - 1, 1);
    const y = top + (1 - Math.min(Math.max(zone.error, 0), 1)) * innerHeight;
    return { x, y, zone };
  });
  const line = points.filter((point): point is NonNullable<typeof point> => point !== null);
  const label = zones
    .map((zone) =>
      typeof zone.error === 'number' ? `${zone.label} error ${zone.error.toFixed(2)}` : `${zone.label} n/a`,
    )
    .join(', ');

  return (
    <svg
      className="range-chart"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Distance error by range: ${label}`}
    >
      <line x1={left} y1={top + innerHeight} x2={width - right} y2={top + innerHeight} className="range-axis" />
      {line.length > 1 && (
        <polyline
          fill="none"
          stroke="#f0883e"
          strokeWidth="1.5"
          points={line.map((point) => `${point.x},${point.y}`).join(' ')}
        />
      )}
      {points.map((point, index) =>
        point ? (
          <circle key={zones[index].zone} cx={point.x} cy={point.y} r="2.5" fill="#f0883e" />
        ) : null,
      )}
      {zones.map((zone, index) => {
        const x = left + (index * innerWidth) / Math.max(zones.length - 1, 1);
        return (
          <text key={zone.zone} x={x} y={height - 1} textAnchor="middle" className="range-label">
            {zone.label}
          </text>
        );
      })}
    </svg>
  );
}

export const TopBar: React.FC<TopBarProps> = ({
  fps,
  latencyMs,
  miou,
  gpu,
  distanceZones = [],
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
        <span className="hud-text">
          HUD: {fps} FPS | Latency: {latencyMs.toFixed(1)}ms | IoU: {miou} | GPU: {gpu}
        </span>
        {distanceZones.length > 0 && <DistanceErrorChart zones={distanceZones} />}
      </div>
    </header>
  );
};
