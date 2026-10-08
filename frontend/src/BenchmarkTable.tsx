import React from 'react';
import type { BenchmarkScore, BenchmarkTable as BenchmarkTableData, SavedRun, SavedRunSummary } from './types';

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

function formatCell(value: number | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? formatCount(value) : '—';
}

function formatMaybeScore(value: number | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? formatScore(value) : '—';
}

export const BenchmarkTable: React.FC<{
  table: BenchmarkTableData | null;
  runs?: SavedRunSummary[];
  selectedRun?: SavedRun | null;
  onSelectRun?: (id: string) => void;
}> = ({ table, runs = [], selectedRun = null, onSelectRun }) => {
  const held = table?.heldout;
  const heldLine = held
    ? [heldoutText('Monocular', held.monocular), heldoutText('VoxNet', held.voxnet)]
        .filter((part): part is string => Boolean(part))
        .join(' · ')
    : '';

  return (
    <aside className="overlay-card benchmark-card" aria-label="mIoU benchmark">
      <h2>mIoU benchmark</h2>
      {(!table || table.rows.length === 0) && runs.length === 0 && (
        <p className="bench-note">The 1.0 m comparison is not ready yet.</p>
      )}
      {table && table.rows.length > 0 && (
        <>
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
            <th>Band mIoU</th>
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row) => (
            <tr key={row.pipeline}>
              <th scope="row">{row.pipeline}</th>
              <td>{formatCell(row.tp)}</td>
              <td>{formatCell(row.fp)}</td>
              <td>{formatCell(row.fn)}</td>
              <td>{formatMaybeScore(row.iou)}</td>
              <td>{formatMaybeScore(row.miou)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="bench-note">
        IoU = TP / (TP + FP + FN) on exact {table.voxel_m.toFixed(1)} m cells. Unknown cells are
        ignored. Band mIoU averages the height bands below 0.45 m, 0.45 to 2.3 m, and above 2.3 m.
        Those bands are not object classes. HUD IoU is the live 0.2 m score with one-cell tolerance.
      </p>
      {held && heldLine && (
        <p className="bench-note">
          Held-out micro ({held.frames} frames): {heldLine}
        </p>
      )}
        </>
      )}
      {runs.length > 0 && (
        <div className="saved-runs">
          <p className="bench-note">Saved runs</p>
          <ul>
            {runs.map((run) => (
              <li key={run.id}>
                <button
                  type="button"
                  className={selectedRun?.id === run.id ? 'active' : undefined}
                  onClick={() => onSelectRun?.(run.id)}
                >
                  {run.checkpoint_id ?? run.id}
                  {run.heldout ? ' · held-out' : run.frame_index != null ? ` · frame ${run.frame_index}` : ''}
                  {run.rows?.map((row) =>
                    typeof row.iou === 'number' ? ` · ${row.pipeline} ${row.iou.toFixed(3)}` : '',
                  )}
                </button>
              </li>
            ))}
          </ul>
          {selectedRun?.frame?.rows && (
            <table>
              <tbody>
                {selectedRun.frame.rows.map((row) => (
                  <tr key={row.pipeline}>
                    <th scope="row">{row.pipeline}</th>
                    <td>{formatCell(row.tp)}</td>
                    <td>{formatCell(row.fp)}</td>
                    <td>{formatCell(row.fn)}</td>
                    <td>{formatMaybeScore(row.iou)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </aside>
  );
};
