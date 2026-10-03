import React from 'react';
import type { SelectedVoxel } from './types';

interface InspectionPanelProps {
  selected: SelectedVoxel;
  onClose: () => void;
}

const CLASS_LABEL: Record<string, string> = {
  driveable: 'Below 0.45 m',
  vehicle: '0.45 to 2.3 m',
  pedestrian: 'Above 2.3 m',
};

function signed(n: number, digits = 1): string {
  const abs = Math.abs(n).toFixed(digits);
  return `${n >= 0 ? '+' : '-'}${abs}`;
}

export const InspectionPanel: React.FC<InspectionPanelProps> = ({ selected, onClose }) => {
  const { voxel, meters, source, cameras } = selected;
  const band = CLASS_LABEL[voxel.cls] ?? voxel.cls;

  return (
    <aside className="overlay-card inspect-card" role="dialog" aria-label="Voxel inspection">
      <div className="card-head">
        <h2>Voxel Inspection</h2>
        <button type="button" className="close-btn" onClick={onClose} aria-label="Close inspection">
          X
        </button>
      </div>

      <div className="field-label">Coordinates:</div>
      <p>
        X = {signed(meters.x)}m
        <br />
        Y = {signed(meters.y)}m
        <br />
        Z = {signed(meters.z)}m
      </p>

      <div className="field-label">Height band:</div>
      <p>{band}</p>

      <div className="field-label">Source:</div>
      <p>{source}</p>

      <div className="field-label">Cameras:</div>
      <p>{cameras}</p>
    </aside>
  );
};
