import React, { useEffect, useRef, useMemo, type MutableRefObject } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import type { SemanticClass, SelectedVoxel, VoxelData } from './types';
import { HEIGHT_BANDS, HEIGHT_COLOR } from './heightBands';

export type SharedOrbit = {
  position: THREE.Vector3;
  target: THREE.Vector3;
  driver: string | null;
};

export function createSharedOrbit(): SharedOrbit {
  return {
    position: new THREE.Vector3(12, 10, -22),
    target: new THREE.Vector3(0, 0.4, 10),
    driver: null,
  };
}

type PressClaim = { voxel: VoxelData; stamp: number };

interface VoxelCanvasProps {
  voxelsRef: MutableRefObject<VoxelData[]>;
  voxelSize: number;
  layers: Record<SemanticClass, boolean>;
  selected: SelectedVoxel | null;
  onSelect: (voxel: VoxelData | null) => void;
  title?: string;
  subtitle?: string;
  className?: string;
  orbitRef?: MutableRefObject<SharedOrbit>;
  orbitId?: string;
  discrepancyRef?: MutableRefObject<VoxelData[]>;
  discrepancyEnabled?: boolean;
  discrepancyColor?: string;
  freeRef?: MutableRefObject<VoxelData[]>;
  freeEnabled?: boolean;
  unknownRef?: MutableRefObject<VoxelData[]>;
  unknownEnabled?: boolean;
  ghostRef?: MutableRefObject<VoxelData[]>;
  ghostEnabled?: boolean;
  models?: string[];
  activeModel?: string;
  onModelChange?: (model: string) => void;
}

const MAX_INSTANCES = 900;
// A real click drifts a few pixels, and these cubes are only a handful of
// pixels across. Treat that as a click on whatever the press hit. A longer
// move is an orbit drag and should not open or close the inspector.
const CLICK_SLOP_PX = 16;
const dummy = new THREE.Object3D();
const color = new THREE.Color();

function hitInstanceId(event: {
  instanceId?: number | null;
  intersections: Array<{ instanceId?: number | null }>;
}): number | null {
  const id =
    event.intersections.find((hit) => hit.instanceId != null)?.instanceId ?? event.instanceId;
  return id == null ? null : id;
}

const CLASS_COLOR = HEIGHT_COLOR;

const PickGesture: React.FC<{
  claimRef: MutableRefObject<PressClaim | null>;
  onSelect: (voxel: VoxelData | null) => void;
}> = ({ claimRef, onSelect }) => {
  const gl = useThree((state) => state.gl);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  useEffect(() => {
    const canvas = gl.domElement;
    const down = { x: 0, y: 0, voxel: null as VoxelData | null };
    const onDown = (event: PointerEvent) => {
      const claim = claimRef.current;
      down.x = event.clientX;
      down.y = event.clientY;
      // Mesh handlers run in the capture phase and stamp this press.
      down.voxel = claim && claim.stamp === event.timeStamp ? claim.voxel : null;
    };
    const onUp = (event: PointerEvent) => {
      const voxel = down.voxel;
      down.voxel = null;
      const dx = event.clientX - down.x;
      const dy = event.clientY - down.y;
      if (dx * dx + dy * dy > CLICK_SLOP_PX * CLICK_SLOP_PX) return;
      onSelectRef.current(voxel);
    };
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointerup', onUp);
    return () => {
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointerup', onUp);
    };
  }, [claimRef, gl]);

  return null;
};

const VoxelInstances: React.FC<{
  voxelsRef: MutableRefObject<VoxelData[]>;
  voxelSize: number;
  layers: Record<SemanticClass, boolean>;
  selected: VoxelData | null;
  onSelect: (voxel: VoxelData | null) => void;
  claimRef: MutableRefObject<PressClaim | null>;
}> = ({ voxelsRef, voxelSize, layers, selected, onSelect, claimRef }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const visibleRef = useRef<VoxelData[]>([]);
  const layersRef = useRef(layers);
  const sizeRef = useRef(voxelSize);
  const selectedRef = useRef(selected);
  const lastVoxelsRef = useRef<VoxelData[] | null>(null);
  const lastSelectedRef = useRef<VoxelData | null>(null);
  layersRef.current = layers;
  sizeRef.current = voxelSize;
  selectedRef.current = selected;

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const voxelsNow = voxelsRef.current;
    const selectedNow = selectedRef.current;
    const voxelsChanged = voxelsNow !== lastVoxelsRef.current;
    const selectionChanged = selectedNow !== lastSelectedRef.current;
    if (!voxelsChanged && !selectionChanged) return;

    if (voxelsChanged) {
      lastVoxelsRef.current = voxelsNow;
      const layersNow = layersRef.current;
      const sizeNow = sizeRef.current;
      const visible = voxelsNow.filter((voxel) => layersNow[voxel.cls]);
      visibleRef.current = visible;

      const count = Math.min(visible.length, MAX_INSTANCES);
      mesh.count = count;

      for (let i = 0; i < count; i++) {
        const voxel = visible[i];
        // nuScenes ego: x forward, y left, z up → Three.js: x right, y up, z forward
        dummy.position.set(-voxel.y, voxel.z, voxel.x);
        dummy.scale.setScalar(Math.max(sizeNow, 0.12) * 0.9);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }

      mesh.instanceMatrix.needsUpdate = true;
      // InstancedMesh.raycast drops the ray when it misses boundingSphere, and
      // that sphere is filled on the first pointer event — often while every
      // instance is still the identity matrix at the origin. Drawing ignores
      // the sphere (frustumCulled is off), so the cubes stay visible while
      // every click misses and the inspector never opens.
      mesh.computeBoundingSphere();
    }

    lastSelectedRef.current = selectedNow;
    const visible = visibleRef.current;
    for (let i = 0; i < mesh.count; i++) {
      const voxel = visible[i];
      if (!voxel) continue;
      color.set(CLASS_COLOR[voxel.cls]);
      if (
        selectedNow &&
        selectedNow.x === voxel.x &&
        selectedNow.y === voxel.y &&
        selectedNow.z === voxel.z
      ) {
        color.offsetHSL(0, 0, 0.25);
      }
      mesh.setColorAt(i, color);
    }
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true;
    }
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, MAX_INSTANCES]}
      frustumCulled={false}
      onPointerDown={(event) => {
        const id = hitInstanceId(event);
        const voxel = id == null ? null : visibleRef.current[id];
        if (!voxel) return;
        claimRef.current = { voxel, stamp: event.nativeEvent.timeStamp };
      }}
      onClick={(event) => {
        event.stopPropagation();
        const instanceId = hitInstanceId(event);
        if (instanceId == null) return;
        onSelect(visibleRef.current[instanceId] ?? null);
      }}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
};

const DiscrepancyOverlay: React.FC<{
  voxelsRef: MutableRefObject<VoxelData[]>;
  voxelSize: number;
  enabled: boolean;
  colorHex: string;
  selected: VoxelData | null;
  claimRef: MutableRefObject<PressClaim | null>;
  onSelect: (voxel: VoxelData | null) => void;
}> = ({ voxelsRef, voxelSize, enabled, colorHex, selected, claimRef, onSelect }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const visibleRef = useRef<VoxelData[]>([]);
  const sizeRef = useRef(voxelSize);
  const enabledRef = useRef(enabled);
  const colorRef = useRef(colorHex);
  const selectedRef = useRef(selected);
  const lastVoxelsRef = useRef<VoxelData[] | null>(null);
  const lastEnabledRef = useRef(enabled);
  const lastColorRef = useRef(colorHex);
  const lastSelectedRef = useRef<VoxelData | null>(null);
  sizeRef.current = voxelSize;
  enabledRef.current = enabled;
  colorRef.current = colorHex;
  selectedRef.current = selected;

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const voxelsNow = voxelsRef.current;
    const enabledNow = enabledRef.current;
    const colorNow = colorRef.current;
    const selectedNow = selectedRef.current;
    const voxelsChanged = voxelsNow !== lastVoxelsRef.current;
    const enabledChanged = enabledNow !== lastEnabledRef.current;
    const colorChanged = colorNow !== lastColorRef.current;
    const selectionChanged = selectedNow !== lastSelectedRef.current;
    if (!voxelsChanged && !enabledChanged && !colorChanged && !selectionChanged) return;

    lastEnabledRef.current = enabledNow;
    lastColorRef.current = colorNow;
    lastSelectedRef.current = selectedNow;
    if (!enabledNow) {
      mesh.count = 0;
      visibleRef.current = [];
      lastVoxelsRef.current = voxelsNow;
      return;
    }

    if (voxelsChanged || enabledChanged) {
      lastVoxelsRef.current = voxelsNow;
      const visible = voxelsNow;
      visibleRef.current = visible;
      const count = Math.min(visible.length, MAX_INSTANCES);
      mesh.count = count;
      const scale = Math.max(sizeRef.current, 0.12) * 1.02;
      for (let i = 0; i < count; i++) {
        const voxel = visible[i];
        dummy.position.set(-voxel.y, voxel.z, voxel.x);
        dummy.scale.setScalar(scale);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
    }

    const visible = visibleRef.current;
    color.set(colorNow);
    for (let i = 0; i < mesh.count; i++) {
      const voxel = visible[i];
      if (!voxel) continue;
      color.set(colorNow);
      if (
        selectedNow &&
        selectedNow.x === voxel.x &&
        selectedNow.y === voxel.y &&
        selectedNow.z === voxel.z
      ) {
        color.offsetHSL(0, 0, 0.18);
      }
      mesh.setColorAt(i, color);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, MAX_INSTANCES]}
      frustumCulled={false}
      onPointerDown={(event) => {
        const id = hitInstanceId(event);
        const voxel = id == null ? null : visibleRef.current[id];
        if (!voxel) return;
        claimRef.current = { voxel, stamp: event.nativeEvent.timeStamp };
      }}
      onClick={(event) => {
        event.stopPropagation();
        const instanceId = hitInstanceId(event);
        if (instanceId == null) return;
        onSelect(visibleRef.current[instanceId] ?? null);
      }}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
};

const GHOST_COLOR = '#c5d4e0';

const GhostOverlay: React.FC<{
  voxelsRef: MutableRefObject<VoxelData[]>;
  voxelSize: number;
  enabled: boolean;
  layers: Record<SemanticClass, boolean>;
}> = ({ voxelsRef, voxelSize, enabled, layers }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const sizeRef = useRef(voxelSize);
  const enabledRef = useRef(enabled);
  const layersRef = useRef(layers);
  const lastVoxelsRef = useRef<VoxelData[] | null>(null);
  const lastEnabledRef = useRef(enabled);
  const lastLayersRef = useRef(layers);
  const lastSizeRef = useRef(voxelSize);
  sizeRef.current = voxelSize;
  enabledRef.current = enabled;
  layersRef.current = layers;

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const voxelsNow = voxelsRef.current;
    const enabledNow = enabledRef.current;
    const layersNow = layersRef.current;
    const sizeNow = sizeRef.current;
    const voxelsChanged = voxelsNow !== lastVoxelsRef.current;
    const enabledChanged = enabledNow !== lastEnabledRef.current;
    const layersChanged = layersNow !== lastLayersRef.current;
    const sizeChanged = sizeNow !== lastSizeRef.current;
    if (!voxelsChanged && !enabledChanged && !layersChanged && !sizeChanged) return;

    lastEnabledRef.current = enabledNow;
    lastLayersRef.current = layersNow;
    lastVoxelsRef.current = voxelsNow;
    lastSizeRef.current = sizeNow;
    if (!enabledNow) {
      mesh.count = 0;
      return;
    }

    const visible = voxelsNow.filter((voxel) => layersNow[voxel.cls]);
    const count = Math.min(visible.length, MAX_INSTANCES);
    mesh.count = count;
    const scale = Math.max(sizeNow, 0.12) * 1.02;
    color.set(GHOST_COLOR);
    for (let i = 0; i < count; i++) {
      const voxel = visible[i];
      dummy.position.set(-voxel.y, voxel.z, voxel.x);
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, MAX_INSTANCES]}
      frustumCulled={false}
      raycast={() => null}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial
        toneMapped={false}
        transparent
        opacity={0.32}
        depthWrite={false}
        polygonOffset
        polygonOffsetFactor={-1}
      />
    </instancedMesh>
  );
};

const SyncedOrbitControls: React.FC<{
  orbitRef?: MutableRefObject<SharedOrbit>;
  orbitId?: string;
}> = ({ orbitRef, orbitId }) => {
  const controlsRef = useRef<{
    object: THREE.Camera;
    target: THREE.Vector3;
    update: () => void;
  } | null>(null);

  useFrame(() => {
    const controls = controlsRef.current;
    if (!controls || !orbitRef || !orbitId) return;
    const shared = orbitRef.current;
    if (shared.driver === orbitId) {
      shared.position.copy(controls.object.position);
      shared.target.copy(controls.target);
      return;
    }
    if (shared.driver == null) return;
    controls.object.position.copy(shared.position);
    controls.target.copy(shared.target);
    controls.update();
  });

  return (
    <OrbitControls
      ref={controlsRef as never}
      makeDefault
      enableDamping
      dampingFactor={0.05}
      target={[0, 0.4, 10]}
      maxDistance={80}
      onStart={() => {
        if (orbitRef && orbitId) orbitRef.current.driver = orbitId;
      }}
    />
  );
};

export const VoxelCanvas: React.FC<VoxelCanvasProps> = ({
  voxelsRef,
  voxelSize,
  layers,
  selected,
  onSelect,
  title = 'Vision Prediction',
  subtitle = '3D Voxel Grid Scene',
  className = 'pane',
  orbitRef,
  orbitId,
  discrepancyRef,
  discrepancyEnabled = false,
  discrepancyColor = '#f0883e',
  freeRef,
  freeEnabled = false,
  unknownRef,
  unknownEnabled = false,
  ghostRef,
  ghostEnabled = false,
  models,
  activeModel,
  onModelChange,
}) => {
  const claimRef = useRef<PressClaim | null>(null);
  const layerKey = useMemo(
    () => `${layers.driveable}-${layers.vehicle}-${layers.pedestrian}-${voxelSize}`,
    [layers, voxelSize],
  );
  return (
    <section className={className}>
      <div className="pane-header voxel-header">
        <h1 className="pane-title">
          {title}
          <span className="sub">
            {onModelChange && models && models.length > 0 ? (
              <select
                className="model-select"
                aria-label="Prediction model"
                value={activeModel}
                onChange={(event) => onModelChange(event.target.value)}
              >
                {models.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            ) : (
              subtitle
            )}
          </span>
        </h1>
        <div className="height-legend" aria-label="Height colors">
          <span className="height-legend-title">Height</span>
          {HEIGHT_BANDS.map((band) => (
            <span key={band.id} className="height-legend-row">
              <i className="height-swatch" style={{ background: band.color }} />
              {band.label}
            </span>
          ))}
        </div>
      </div>
      <div className="voxel-stage">
        <Canvas camera={{ position: [12, 10, -22], fov: 50 }}>
          <color attach="background" args={['#0d1117']} />
          <ambientLight intensity={0.8} />

          <PickGesture claimRef={claimRef} onSelect={onSelect} />

          <VoxelInstances
            key={layerKey}
            voxelsRef={voxelsRef}
            voxelSize={voxelSize}
            layers={layers}
            selected={selected?.voxel ?? null}
            onSelect={onSelect}
            claimRef={claimRef}
          />

          {ghostRef && (
            <GhostOverlay
              voxelsRef={ghostRef}
              voxelSize={voxelSize}
              enabled={ghostEnabled}
              layers={layers}
            />
          )}

          {discrepancyRef && (
            <DiscrepancyOverlay
              voxelsRef={discrepancyRef}
              voxelSize={voxelSize}
              enabled={discrepancyEnabled}
              colorHex={discrepancyColor}
              selected={selected?.voxel ?? null}
              claimRef={claimRef}
              onSelect={onSelect}
            />
          )}
          {freeRef && (
            <DiscrepancyOverlay
              voxelsRef={freeRef}
              voxelSize={voxelSize}
              enabled={freeEnabled}
              colorHex="#9aa4b2"
              selected={selected?.voxel ?? null}
              claimRef={claimRef}
              onSelect={onSelect}
            />
          )}
          {unknownRef && (
            <DiscrepancyOverlay
              voxelsRef={unknownRef}
              voxelSize={voxelSize}
              enabled={unknownEnabled}
              colorHex="#3d4450"
              selected={selected?.voxel ?? null}
              claimRef={claimRef}
              onSelect={onSelect}
            />
          )}

          {/* The grid shader lays the plane flat, but raycasts still see the
              upright quad, so it sat through the cloud and swallowed clicks. */}
          <Grid
            raycast={() => null}
            position={[0, 0, 12]}
            args={[50, 50]}
            cellSize={1}
            cellThickness={0.6}
            cellColor="#30363d"
            sectionSize={5}
            sectionThickness={1.2}
            sectionColor="#58a6ff"
            fadeDistance={80}
          />

          <SyncedOrbitControls orbitRef={orbitRef} orbitId={orbitId} />
        </Canvas>
      </div>
    </section>
  );
};
