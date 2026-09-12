import React from 'react';
import playPauseIcon from './assets/playpause.svg';

interface PlaybackBarProps {
  playing: boolean;
  onTogglePlay: () => void;
  frameIndex: number;
  frameCount: number;
  durationS: number;
  sampleHz: number;
  onSeek: (frame: number) => void;
}

function formatTime(seconds: number): string {
  const clamped = Math.max(0, seconds);
  const m = Math.floor(clamped / 60);
  const s = clamped - m * 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

export const PlaybackBar: React.FC<PlaybackBarProps> = ({
  playing,
  onTogglePlay,
  frameIndex,
  frameCount,
  durationS,
  sampleHz,
  onSeek,
}) => {
  const currentS = frameCount > 1
    ? (frameIndex / Math.max(frameCount - 1, 1)) * durationS
    : frameIndex / Math.max(sampleHz, 1);
  const maxIndex = Math.max(frameCount - 1, 0);

  return (
    <footer className="playback-bar">
      <div className="playback-label">PLAYBACK:</div>
      <button
        type="button"
        className="play-btn"
        onClick={onTogglePlay}
        aria-label={playing ? 'Pause playback' : 'Play playback'}
      >
        <img src={playPauseIcon} alt="" />
      </button>
      <input
        className="scrubber"
        type="range"
        min={0}
        max={maxIndex}
        step={1}
        value={Math.min(frameIndex, maxIndex)}
        onChange={(e) => onSeek(Number(e.target.value))}
        aria-label="Playback scrubber"
      />
      <div className="timecode">
        {formatTime(currentS)} / {formatTime(durationS)}
      </div>
    </footer>
  );
};
