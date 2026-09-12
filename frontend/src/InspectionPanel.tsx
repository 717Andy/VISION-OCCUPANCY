import React from 'react';
import type { SelectedVoxel } from './types';

interface InspectionPanelProps {
  selected: SelectedVoxel;
  onClose: () => void;
}

const CLASS_LABEL: Record<string, string> = {
  driveable: 'Driveable Area',
  vehicle: 'Vehicle',
  pedestrian: 'Pedestrian',
};

function signed(n: number, digits = 1): string {
  const abs = Math.abs(n).toFixed(digits);
  return `${n >= 0 ? '+' : '-'}${abs}`;
}

export const InspectionPanel: React.FC<InspectionPanelProps> = ({ selected, onClose }) => {
  const { voxel, meters, depthSource, flow } = selected;
  const primary = CLASS_LABEL[voxel.cls] ?? voxel.cls;

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

      <div className="field-label">Primary Class:</div>
      <p>
        {primary} (Confidence: {(voxel.prob * 100).toFixed(1)}%)
      </p>

      <div className="field-label">Secondary Probabilities:</div>
      <p>
        Obstacle ({Math.max(0.4, (1 - voxel.prob) * 70).toFixed(1)}%)
        <br />
        Free Space ({Math.max(0.3, (1 - voxel.prob) * 30).toFixed(1)}%)
      </p>

      <div className="field-label">Flow Vector:</div>
      <p>
        Vx = {signed(flow.vx)} m/s
        <br />
        Vy = {signed(flow.vy)} m/s
      </p>

      <div className="field-label">Depth Source:</div>
      <p>{depthSource}</p>
    </aside>
  );
};
