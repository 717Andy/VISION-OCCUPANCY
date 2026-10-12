import React, { useEffect, useState } from 'react';
import { fetchCrossScene } from './api';
import type {
  BenchmarkTable as BenchmarkTableData,
  CrossSceneProtocol,
  SavedRun,
  SavedRunSummary,
} from './types';

function formatCount(value: number): string {
  return Math.round(value).toLocaleString('en-US');
}

function formatScore(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : '—';
}

function heldoutText(label: string, score: { iou?: number; miou?: number } | undefined): string | null {
  if (!score || typeof score.iou !== 'number') return null;
  const miou = typeof score.miou === 'number' ? score.miou : score.iou;
  return `${label} IoU ${formatScore(score.iou)} mIoU ${formatScore(miou)}`;
}

const DEFAULT_COMPARE = ['VoxNet 3D CNN', 'Lift-Splat'];

function deltaText(iou: number | undefined, reference: number | undefined): string {
  if (typeof iou !== 'number' || typeof reference !== 'number') return '—';
  const delta = iou - reference;
  return `${delta >= 0 ? '+' : ''}${delta.toFixed(3)}`;
}

function formatCell(value: number | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? formatCount(value) : '—';
}

function heldScoresFor(
  held: BenchmarkTableData['heldout'],
  compared: string[],
): { pipeline: string; iou?: number; miou?: number }[] {
  if (!held) return [];
  const fromScores = (held.scores ?? []).filter((row) => compared.includes(row.pipeline) && !row.error);
  if (fromScores.length > 0) return fromScores;
  const legacy: { pipeline: string; iou?: number; miou?: number }[] = [];
  if (held.monocular) legacy.push({ pipeline: 'Monocular depth', iou: held.monocular.iou, miou: held.monocular.miou });
  if (held.voxnet) legacy.push({ pipeline: 'VoxNet 3D CNN', iou: held.voxnet.iou, miou: held.voxnet.miou });
  return legacy.filter((row) => compared.includes(row.pipeline));
}

function formatMaybeScore(value: number | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? formatScore(value) : '—';
}

export const BenchmarkTable: React.FC<{
  table: BenchmarkTableData | null;
  runs?: SavedRunSummary[];
  selectedRun?: SavedRun | null;
  onSelectRun?: (id: string) => void;
  viewingModel?: string;
  scene?: string;
  frameIndex?: number;
}> = ({ table, runs = [], selectedRun = null, onSelectRun, viewingModel, scene, frameIndex }) => {
  const [chosen, setChosen] = useState<string[]>(DEFAULT_COMPARE);
  const [reference, setReference] = useState('Lift-Splat');
  const [crossScene, setCrossScene] = useState<CrossSceneProtocol | null>(null);

  useEffect(() => {
    fetchCrossScene().then(setCrossScene).catch(() => setCrossScene(null));
  }, []);
  const names = table?.rows.map((row) => row.pipeline) ?? [];
  const active = chosen.filter((name) => names.includes(name));
  const compared = active.length >= 2 ? active : names.slice(0, Math.min(2, names.length));
  const referenceName = compared.includes(reference) ? reference : compared[0];
  const visible = (table?.rows ?? []).filter((row) => compared.includes(row.pipeline));
  const referenceRow = visible.find((row) => row.pipeline === referenceName);
  const viewingRow = (table?.rows ?? []).find((row) => row.pipeline === viewingModel);
  const viewingCompared = Boolean(viewingModel && compared.includes(viewingModel));
  const viewingScore =
    viewingRow && typeof viewingRow.iou === 'number' && Number.isFinite(viewingRow.iou)
      ? viewingRow.iou.toFixed(3)
      : null;
  const frameReady = table != null && (frameIndex == null || table.frame_index === frameIndex);
  const protocolTitle = `${(table?.voxel_m ?? 1).toFixed(1)} m protocol · exact cells · unknown ignored`;
  const held = table?.heldout;
  const heldScores = heldScoresFor(held, compared);
  const heldLine = heldScores
    .map((row) => heldoutText(row.pipeline, row))
    .filter((part): part is string => Boolean(part))
    .join(' · ');

  const toggle = (name: string) => {
    setChosen((current) => {
      const present = current.filter((item) => names.includes(item));
      const base = present.length >= 2 ? present : names.slice(0, Math.min(2, names.length));
      if (base.includes(name)) {
        if (base.length <= 2) return base;
        return base.filter((item) => item !== name);
      }
      return [...base, name];
    });
  };

  return (
    <aside className="overlay-card benchmark-card" aria-label="1.0 m protocol">
      <h2>{protocolTitle}</h2>
      {(!table || table.rows.length === 0) && runs.length === 0 && (
        <p className="bench-note">The 1.0 m comparison is not ready yet.</p>
      )}
      {table && table.rows.length > 0 && (
        <>
      {frameReady ? (
        <p className="bench-meta">
          Frame {table.frame_index} · {table.split}
        </p>
      ) : (
        <p className="bench-note">Scoring frame {frameIndex ?? table.frame_index}…</p>
      )}
      {frameReady && viewingModel && !viewingCompared && (
        <p className="bench-note viewing-chip">
          Now viewing {viewingModel} · protocol IoU {viewingScore ?? '—'}
        </p>
      )}
      <div className="bench-pick" role="group" aria-label="Models to compare">
        {table.rows.map((row) => (
          <div key={row.pipeline} className="bench-choice">
            <input
              type="checkbox"
              checked={compared.includes(row.pipeline)}
              aria-label={`Compare ${row.pipeline}`}
              onChange={() => toggle(row.pipeline)}
            />
            <input
              type="radio"
              name="benchmark-reference"
              checked={referenceName === row.pipeline}
              disabled={!compared.includes(row.pipeline)}
              onChange={() => setReference(row.pipeline)}
              aria-label={`${row.pipeline} reference`}
            />
            <span>{row.pipeline}</span>
          </div>
        ))}
      </div>
      {crossScene && crossScene.scenes.length > 0 && (
        <section className="cross-scene" aria-label="Protocol IoU across clips">
          <h3>Across clips</h3>
          <table>
            <thead>
              <tr>
                <th>Pipeline</th>
                {crossScene.scenes.map((column) => (
                  <th key={column.name} className={column.name === scene ? 'current' : undefined} scope="col">
                    <span className="cross-heading">{column.heading}</span>
                    <span className="cross-detail">{column.detail}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {compared.map((name) => {
                const values = crossScene.scenes.map((column) => {
                  const score = column.scores.find((row) => row.pipeline === name && !row.error);
                  return typeof score?.iou === 'number' ? score.iou : null;
                });
                const finite = values.filter((value): value is number => value !== null);
                const best = finite.length > 0 ? Math.max(...finite) : null;
                return (
                  <tr key={name} className={name === viewingModel ? 'viewing' : undefined}>
                    <th scope="row">
                      {name}
                      {name === viewingModel && <span className="viewing-badge">viewing</span>}
                    </th>
                    {crossScene.scenes.map((column, index) => {
                      const value = values[index];
                      const isBest = value !== null && best !== null && value === best;
                      return (
                        <td
                          key={column.name}
                          className={[column.name === scene ? 'current' : '', isBest ? 'best' : ''].filter(Boolean).join(' ') || undefined}
                        >
                          {value === null ? '—' : formatScore(value)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="bench-note">
            IoU on the {(crossScene.voxel_m ?? 1).toFixed(1)} m protocol. {crossScene.scenes.find((column) => column.training_scene)?.heading ?? '0103'} uses its held-out frames. The other clips use every frame.
          </p>
        </section>
      )}
      {frameReady && <table>
        <thead>
          <tr>
            <th>Pipeline</th>
            <th>TP</th>
            <th>FP</th>
            <th>FN</th>
            <th>Prec</th>
            <th>Rec</th>
            <th>IoU</th>
            <th>ΔIoU</th>
            <th>Band mIoU</th>
            <th>ms</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((row) => (
            <tr key={row.pipeline} className={row.pipeline === viewingModel ? 'viewing' : undefined}>
              <th scope="row">
                {row.pipeline}
                {row.pipeline === viewingModel && <span className="viewing-badge">viewing</span>}
              </th>
              <td>{formatCell(row.tp)}</td>
              <td>{formatCell(row.fp)}</td>
              <td>{formatCell(row.fn)}</td>
              <td>{formatMaybeScore(row.precision ?? undefined)}</td>
              <td>{formatMaybeScore(row.recall ?? undefined)}</td>
              <td>{formatMaybeScore(row.iou)}</td>
              <td>{deltaText(row.iou, referenceRow?.iou)}</td>
              <td>{formatMaybeScore(row.miou)}</td>
              <td>{typeof row.latency_ms === 'number' ? Math.round(row.latency_ms).toLocaleString('en-US') : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>}
      <p className="bench-note">
        IoU = TP / (TP + FP + FN) on exact {table.voxel_m.toFixed(1)} m cells. Unknown cells are
        ignored. Band mIoU averages the height bands below 0.45 m, 0.45 to 2.3 m, and above 2.3 m.
        Those bands are not object classes. Live IoU is the on-screen score at the live voxel size with one-cell tolerance.
      </p>
      {held && heldLine && (
        <p className="bench-note">
          {held.label || 'Held-out'} micro ({held.frames} frames): {heldLine}
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
