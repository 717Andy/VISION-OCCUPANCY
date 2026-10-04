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

export interface BenchmarkBandCount {
  band: string;
  tp: number;
  fp: number;
  fn: number;
  iou: number;
}

export interface BenchmarkRow {
  pipeline: string;
  tp: number;
  fp: number;
  fn: number;
  iou: number;
  miou: number;
  bands?: BenchmarkBandCount[];
}

export interface BenchmarkScore {
  tp: number;
  fp: number;
  fn: number;
  iou: number;
  miou: number;
}

export interface BenchmarkTable {
  voxel_m: number;
  tolerance_cells?: number;
  unknown?: string;
  frame_index: number;
  split: string;
  monocular_threshold: number;
  voxnet_threshold: number;
  checkpoint_id?: string;
  scoring?: string;
  rows: BenchmarkRow[];
  heldout?: {
    frames: number;
    monocular_threshold: number;
    monocular?: BenchmarkScore;
    voxnet?: BenchmarkScore;
  } | null;
}

export interface SavedRunSummary {
  id: string;
  scene_name?: string;
  checkpoint_id?: string;
  heldout?: boolean;
  frame_index?: number;
  rows?: { pipeline?: string; iou?: number; tp?: number; fp?: number; fn?: number }[];
}

export interface SavedRun {
  id: string;
  scene_name?: string;
  checkpoint_id?: string;
  scoring?: string;
  frame?: BenchmarkTable;
  frames?: BenchmarkTable[];
}

export interface SelectedVoxel {
  voxel: VoxelData;
  meters: { x: number; y: number; z: number };
  source: string;
  cameras: string;
}
