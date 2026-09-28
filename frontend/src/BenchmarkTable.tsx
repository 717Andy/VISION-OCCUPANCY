import React from 'react';
import type { BenchmarkScore, BenchmarkTable as BenchmarkTableData } from './types';

function formatCount(value: number): string {
  return Math.round(value).toLocaleString('en-US');
}

function formatScore(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : '—';
}

function heldoutText(label: string, score: BenchmarkScore | undefined): string | null {
  if (!score) return null;
  return `${label} IoU ${formatScore(score.iou)} mIoU ${formatScore(score.miou)}`;
}

export const BenchmarkTable: React.FC<{ table: BenchmarkTableData | null }> = ({ table }) => {
  if (!table || table.rows.length === 0) return null;
  const held = table.heldout;
  const heldLine = held
    ? [heldoutText('Monocular', held.monocular), heldoutText('VoxNet', held.voxnet)]
        .filter((part): part is string => Boolean(part))
        .join(' · ')
    : '';

  return (
    <aside className="overlay-card benchmark-card" aria-label="mIoU benchmark">
      <h2>mIoU benchmark</h2>
      <p className="bench-meta">
        Frame {table.frame_index} · {table.split} · {table.voxel_m.toFixed(1)} m cells
      </p>
      <table>
        <thead>
          <tr>
            <th>Pipeline</th>
            <th>TP</th>
            <th>FP</th>
            <th>FN</th>
            <th>IoU</th>
            <th>mIoU</th>
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row) => (
            <tr key={row.pipeline}>
              <th scope="row">{row.pipeline}</th>
              <td>{formatCount(row.tp)}</td>
              <td>{formatCount(row.fp)}</td>
              <td>{formatCount(row.fn)}</td>
              <td>{formatScore(row.iou)}</td>
              <td>{formatScore(row.miou)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="bench-note">
        IoU = TP / (TP + FP + FN) on exact {table.voxel_m.toFixed(1)} m cells. mIoU averages the
        driveable, vehicle, and pedestrian height bands. HUD mIoU stays the live 0.2 m score with
        one-cell tolerance.
      </p>
      {held && heldLine && (
        <p className="bench-note">
          Held-out micro ({held.frames} frames): {heldLine}
        </p>
      )}
    </aside>
  );
};
