import React, { useRef, useMemo, type MutableRefObject } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import type { SemanticClass, SelectedVoxel, VoxelData } from './types';

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
}

const MAX_INSTANCES = 900;
const dummy = new THREE.Object3D();
const color = new THREE.Color();

const CLASS_COLOR: Record<SemanticClass, string> = {
  driveable: '#7ee787',
  vehicle: '#58a6ff',
  pedestrian: '#f85149',
};

const VoxelInstances: React.FC<{
  voxelsRef: MutableRefObject<VoxelData[]>;
  voxelSize: number;
  layers: Record<SemanticClass, boolean>;
  selected: VoxelData | null;
  onSelect: (voxel: VoxelData | null) => void;
}> = ({ voxelsRef, voxelSize, layers, selected, onSelect }) => {
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
      onClick={(event) => {
        event.stopPropagation();
        // The native click event is spread on top of the hit and can wipe
        // instanceId. The intersection list is copied before that spread.
        const instanceId =
          event.intersections.find((hit) => hit.instanceId != null)?.instanceId ??
          event.instanceId;
        if (instanceId == null) return;
        onSelect(visibleRef.current[instanceId] ?? null);
      }}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial toneMapped={false} />
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
}) => {
  const layerKey = useMemo(
    () => `${layers.driveable}-${layers.vehicle}-${layers.pedestrian}-${voxelSize}`,
    [layers, voxelSize],
  );

  return (
    <section className={className}>
      <div className="pane-header">
        <h1 className="pane-title">
          {title}
          <span className="sub">{subtitle}</span>
        </h1>
      </div>
      <div className="voxel-stage">
        <Canvas
          camera={{ position: [12, 10, -22], fov: 50 }}
          onPointerMissed={() => onSelect(null)}
        >
          <color attach="background" args={['#0d1117']} />
          <ambientLight intensity={0.8} />

          <VoxelInstances
            key={layerKey}
            voxelsRef={voxelsRef}
            voxelSize={voxelSize}
            layers={layers}
            selected={selected?.voxel ?? null}
            onSelect={onSelect}
          />

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
