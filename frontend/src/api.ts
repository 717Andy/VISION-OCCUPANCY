import type { CrossSceneProtocol, SavedRun, SavedRunSummary, SceneManifest } from './types';

export const API_BASE = '/api';

export function wsUrl(path: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}${path}`;
}

export async function fetchScenes(): Promise<{
  name: string;
  description?: string;
  location?: string;
  training_scene?: boolean;
}[]> {
  const res = await fetch(`${API_BASE}/scenes`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.scenes ?? [];
}

export async function fetchManifest(scene?: string): Promise<SceneManifest | null> {
  const qs = scene ? `?scene=${encodeURIComponent(scene)}` : '';
  const res = await fetch(`${API_BASE}/scene${qs}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Failed to load scene manifest (${res.status})`);
  return res.json();
}

export function cameraFrameUrl(frameIndex: number, cameraId: string, scene?: string, cacheBust?: number): string {
  const params = new URLSearchParams();
  if (scene) params.set('scene', scene);
  if (cacheBust != null) params.set('t', String(cacheBust));
  const qs = params.toString();
  return `${API_BASE}/frames/${frameIndex}/cameras/${encodeURIComponent(cameraId)}${qs ? `?${qs}` : ''}`;
}

export async function fetchCrossScene(): Promise<CrossSceneProtocol | null> {
  const res = await fetch(`${API_BASE}/protocol/scenes`);
  if (!res.ok) return null;
  return res.json();
}

export async function fetchRuns(): Promise<SavedRunSummary[]> {
  const res = await fetch(`${API_BASE}/runs`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.runs ?? [];
}

export async function fetchRun(runId: string): Promise<SavedRun | null> {
  const res = await fetch(`${API_BASE}/runs/${encodeURIComponent(runId)}`);
  if (!res.ok) return null;
  return res.json();
}

export async function fetchModels(): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.models ?? []).map((row: { name?: string }) => row.name).filter(Boolean);
  } catch {
    return [];
  }
}

export async function fetchGpuLabel(): Promise<string> {
  try {
    const res = await fetch(`${API_BASE}/telemetry`);
    if (!res.ok) return 'CPU';
    const data = await res.json();
    const gpu = data.gpu ?? 'CPU';
    const source = data.voxel_source ? ` · ${data.voxel_source}` : '';
    return `${gpu}${source}`;
  } catch {
    return 'CPU';
  }
}
