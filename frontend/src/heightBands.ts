import type { SemanticClass } from './types';

/** One hue, darker near the ground and lighter as the cell rises. */
export const HEIGHT_BANDS: { id: SemanticClass; label: string; color: string }[] = [
  { id: 'driveable', label: 'Below 0.45 m', color: '#1b4f6b' },
  { id: 'vehicle', label: '0.45 to 2.3 m', color: '#3d8eb8' },
  { id: 'pedestrian', label: 'Above 2.3 m', color: '#d7f0fa' },
];

export const HEIGHT_COLOR: Record<SemanticClass, string> = {
  driveable: HEIGHT_BANDS[0].color,
  vehicle: HEIGHT_BANDS[1].color,
  pedestrian: HEIGHT_BANDS[2].color,
};
