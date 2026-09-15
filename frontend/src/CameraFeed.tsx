import React from 'react';
import { cameraFrameUrl } from './api';
import type { SceneManifest } from './types';
import settingsIcon from './assets/settings.svg';

const LAYOUT: { id: string; className: string; fallback: string }[] = [
  { id: 'CAM_FRONT', className: 'cam-front', fallback: 'Front' },
  { id: 'CAM_FRONT_LEFT', className: 'cam-fl', fallback: 'Front Left' },
  { id: 'CAM_FRONT_RIGHT', className: 'cam-fr', fallback: 'Front Right' },
  { id: 'CAM_BACK_LEFT', className: 'cam-bl', fallback: 'Back Left' },
  { id: 'CAM_BACK_RIGHT', className: 'cam-br', fallback: 'Back Right' },
  { id: 'CAM_BACK', className: 'cam-back', fallback: 'Back' },
];

interface CameraFeedProps {
  manifest: SceneManifest | null;
  frameIndex: number;
  settingsOpen: boolean;
  onToggleSettings: () => void;
  highlightCamera?: string | null;
}

function formatSceneTime(timeS: number): string {
  const clamped = Math.max(0, timeS);
  const m = Math.floor(clamped / 60);
  const s = clamped - m * 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

function formatCamOffset(sampleTs: number, camTs: number): string {
  const dtMs = (camTs - sampleTs) / 1000;
  const sign = dtMs >= 0 ? '+' : '';
  return `${sign}${dtMs.toFixed(1)}ms`;
}

export const CameraFeed: React.FC<CameraFeedProps> = ({
  manifest,
  frameIndex,
  settingsOpen,
  onToggleSettings,
  highlightCamera,
}) => {
  const frame = manifest?.frames[frameIndex];
  const frameCount = manifest?.frame_count ?? 0;
  const timeS = frame?.time_s ?? 0;

  return (
    <section className="pane pane-camera">
      <div className="pane-header">
        <button
          type="button"
          className="settings-btn"
          aria-label="Open controls"
          aria-pressed={settingsOpen}
          onClick={onToggleSettings}
        >
          <img src={settingsIcon} alt="" />
        </button>
        <h1 className="pane-title">
          Camera Feed
          <span className="sub">
            {manifest
              ? `t=${formatSceneTime(timeS)} · frame ${frameIndex + 1}/${Math.max(frameCount, 1)}`
              : 'waiting for clip'}
          </span>
        </h1>
      </div>
      <div className="camera-stage">
        <div className="camera-grid">
          {LAYOUT.map((cam) => {
            const path = frame?.cameras[cam.id];
            const camTs = frame?.calibration?.[cam.id]?.timestamp;
            return (
              <article
                key={`${cam.id}-${frameIndex}`}
                className={`cam-tile ${cam.className} ${highlightCamera === cam.id ? 'active' : ''}`}
              >
                {path ? (
                  <img
                    src={cameraFrameUrl(frameIndex, cam.id)}
                    alt={`${cam.fallback} at ${formatSceneTime(timeS)}`}
                    draggable={false}
                  />
                ) : (
                  <div className="cam-placeholder" />
                )}
                <span className="cam-label">
                  {cam.fallback}
                  {camTs != null && frame?.timestamp != null && (
                    <span className="cam-ts">{formatCamOffset(frame.timestamp, camTs)}</span>
                  )}
                </span>
              </article>
            );
          })}
        </div>
        {!manifest && (
          <p className="status-note" style={{ position: 'absolute', bottom: 8, width: '100%' }}>
            Waiting for nuScenes clip…
          </p>
        )}
      </div>
    </section>
  );
};
