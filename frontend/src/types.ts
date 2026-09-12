export type SemanticClass = 'driveable' | 'vehicle' | 'pedestrian';

export interface VoxelData {
  x: number;
  y: number;
  z: number;
  prob: number;
  cls: SemanticClass;
}

export interface CameraMeta {
  id: string;
  label: string;
}

export interface SceneFrame {
  index: number;
  timestamp: number;
  cameras: Record<string, string>;
}

export interface SceneManifest {
  scene_name: string;
  description: string;
  location: string;
  duration_s: number;
  sample_hz: number;
  frame_count: number;
  cameras: CameraMeta[];
  frames: SceneFrame[];
}

export interface ControlsState {
  voxelSize: number;
  threshold: number;
  layers: Record<SemanticClass, boolean>;
}

export interface SelectedVoxel {
  voxel: VoxelData;
  meters: { x: number; y: number; z: number };
  depthSource: string;
  flow: { vx: number; vy: number };
}
