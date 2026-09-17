export type SemanticClass = 'driveable' | 'vehicle' | 'pedestrian';

export type ViewMode = 'single' | 'split';

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

export interface CameraCalibration {
  intrinsic: number[][];
  translation: number[];
  rotation: number[];
  timestamp: number;
}

export interface SceneFrame {
  index: number;
  timestamp: number;
  time_s?: number;
  cameras: Record<string, string>;
  calibration?: Record<string, CameraCalibration>;
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
  original_image_size?: [number, number];
  prepared_image_size?: [number, number];
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
