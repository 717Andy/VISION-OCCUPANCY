import React, { useState } from 'react';
import { BenchmarkTable } from './BenchmarkTable';
import type {
  BenchmarkTable as BenchmarkTableData,
  DistanceZone,
  MiouDrop,
  SavedRun,
  SavedRunSummary,
} from './types';

function formatDrop(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '—';
}

function DistanceErrorChart({ zones }: { zones: DistanceZone[] }) {
  const width = 400;
  const height = 220;
  const left = 36;
  const right = 16;
  const top = 22;
  const bottom = 46;
  const innerWidth = width - left - right;
  const innerHeight = height - top - bottom;
  const points = zones.map((zone, index) => {
    if (typeof zone.error !== 'number' || !Number.isFinite(zone.error)) return null;
    const error = zone.error;
    const x = left + (index * innerWidth) / Math.max(zones.length - 1, 1);
    const y = top + (1 - Math.min(Math.max(error, 0), 1)) * innerHeight;
    return { x, y, error };
  });
  const line = points.filter((point): point is NonNullable<typeof point> => point !== null);
  const label = zones
    .map((zone) =>
      typeof zone.error === 'number' ? `${zone.label} error ${zone.error.toFixed(3)}` : `${zone.label} n/a`,
    )
    .join(', ');

  return (
    <svg
      className="range-chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Distance error by range: ${label}`}
    >
      {[0, 0.5, 1].map((tick) => {
        const y = top + (1 - tick) * innerHeight;
        return (
          <g key={tick}>
            <line x1={left} y1={y} x2={width - right} y2={y} className="range-grid" />
            <text x={left - 8} y={y + 4} textAnchor="end" className="range-label">
              {tick.toFixed(1)}
            </text>
          </g>
        );
      })}
      {line.length > 1 && (
        <polyline
          fill="none"
          stroke="#f0883e"
          strokeWidth="2.5"
          points={line.map((point) => `${point.x},${point.y}`).join(' ')}
        />
      )}
      {points.map((point, index) =>
        point ? (
          <g key={zones[index].zone}>
            <circle cx={point.x} cy={point.y} r="4.5" fill="#f0883e" />
            <text x={point.x} y={point.y - 10} textAnchor="middle" className="range-value">
              {point.error.toFixed(2)}
            </text>
          </g>
        ) : null,
      )}
      {zones.map((zone, index) => {
        const x = left + (index * innerWidth) / Math.max(zones.length - 1, 1);
        const miou =
          typeof zone.miou === 'number' && Number.isFinite(zone.miou) ? zone.miou.toFixed(3) : '—';
        return (
          <g key={zone.zone}>
            <text x={x} y={height - 22} textAnchor="middle" className="range-label">
              {zone.label}
            </text>
            <text x={x} y={height - 6} textAnchor="middle" className="range-sublabel">
              mIoU {miou}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function TableIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="4" width="18" height="16" rx="2" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M3 9h18M3 14h18M9 9v11" fill="none" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function ChartIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 19V5M4 19h16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path
        d="M7 15l4-4 3 2 5-6"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export const CornerDock: React.FC<{
  table: BenchmarkTableData | null;
  runs: SavedRunSummary[];
  selectedRun: SavedRun | null;
  onSelectRun: (id: string) => void;
  distanceZones: DistanceZone[];
  miouDrop: MiouDrop;
}> = ({ table, runs, selectedRun, onSelectRun, distanceZones, miouDrop }) => {
  const [benchmarkOpen, setBenchmarkOpen] = useState(false);
  const [chartOpen, setChartOpen] = useState(false);

  return (
    <div className="corner-dock">
      {(benchmarkOpen || chartOpen) && (
        <div className="corner-panels">
          {benchmarkOpen && (
            <BenchmarkTable
              table={table}
              runs={runs}
              selectedRun={selectedRun}
              onSelectRun={onSelectRun}
            />
          )}
          {chartOpen && (
            <aside className="overlay-card range-card" aria-label="Distance decay">
              <h2>Distance decay</h2>
              {distanceZones.length === 0 ? (
                <p className="bench-note">Distance error is not ready yet.</p>
              ) : (
                <>
                  <p className="bench-note">Error = 1 − height-band mIoU. Higher on the chart is a larger miss.</p>
                  <DistanceErrorChart zones={distanceZones} />
                  <p className="bench-note">
                    mIoU drop {formatDrop(miouDrop.near_to_mid)} from 0–10 m to 10–25 m,{' '}
                    {formatDrop(miouDrop.mid_to_far)} from 10–25 m to 25 m+,{' '}
                    {formatDrop(miouDrop.near_to_far)} from 0–10 m to 25 m+.
                  </p>
                </>
              )}
            </aside>
          )}
        </div>
      )}
      <div className="corner-actions">
        <button
          type="button"
          className="dock-btn"
          aria-label="mIoU benchmark"
          aria-pressed={benchmarkOpen}
          onClick={() => setBenchmarkOpen((open) => !open)}
        >
          <TableIcon />
        </button>
        <button
          type="button"
          className="dock-btn"
          aria-label="Distance decay"
          aria-pressed={chartOpen}
          onClick={() => setChartOpen((open) => !open)}
        >
          <ChartIcon />
        </button>
      </div>
    </div>
  );
};
