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

export const CameraFeed: React.FC<CameraFeedProps> = ({
  manifest,
  frameIndex,
  settingsOpen,
  onToggleSettings,
  highlightCamera,
}) => {
  const frame = manifest?.frames[frameIndex];

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
        <h1 className="pane-title">Camera Feed</h1>
      </div>
      <div className="camera-stage">
        <div className="camera-grid">
          {LAYOUT.map((cam) => {
            const path = frame?.cameras[cam.id];
            return (
              <article
                key={cam.id}
                className={`cam-tile ${cam.className} ${highlightCamera === cam.id ? 'active' : ''}`}
              >
                {path ? (
                  <img
                    src={cameraFrameUrl(frameIndex, cam.id)}
                    alt={cam.fallback}
                    draggable={false}
                  />
                ) : (
                  <div className="cam-placeholder" />
                )}
                <span className="cam-label">{cam.fallback}</span>
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
