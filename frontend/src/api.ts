import type { SavedRun, SavedRunSummary, SceneManifest } from './types';

export const API_BASE = '/api';

export function wsUrl(path: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}${path}`;
}

export async function fetchManifest(): Promise<SceneManifest | null> {
  const res = await fetch(`${API_BASE}/scene`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Failed to load scene manifest (${res.status})`);
  return res.json();
}

export function cameraFrameUrl(frameIndex: number, cameraId: string, cacheBust?: number): string {
  const qs = cacheBust != null ? `?t=${cacheBust}` : '';
  return `${API_BASE}/frames/${frameIndex}/cameras/${encodeURIComponent(cameraId)}${qs}`;
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
