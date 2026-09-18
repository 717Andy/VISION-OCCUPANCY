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
    // CAM_FRONT in ego ≈ (1.72, 0, 1.49); Three.js is (-y, z, x)
    position: new THREE.Vector3(0, 1.55, 1.85),
    target: new THREE.Vector3(0, 1.05, 16),
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

const MAX_INSTANCES = 1600;
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
  layersRef.current = layers;
  sizeRef.current = voxelSize;
  selectedRef.current = selected;

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    if (voxelsRef.current === lastVoxelsRef.current) return;
    lastVoxelsRef.current = voxelsRef.current;

    const layersNow = layersRef.current;
    const sizeNow = sizeRef.current;
    const selectedNow = selectedRef.current;
    const visible = voxelsRef.current.filter((voxel) => layersNow[voxel.cls]);
    visibleRef.current = visible;

    const count = Math.min(visible.length, MAX_INSTANCES);
    mesh.count = count;
    if (!mesh.instanceColor) {
      mesh.instanceColor = new THREE.InstancedBufferAttribute(
        new Float32Array(MAX_INSTANCES * 3),
        3,
      );
    }

    for (let i = 0; i < count; i++) {
      const voxel = visible[i];
      // nuScenes ego: x forward, y left, z up → Three.js: x right, y up, z forward
      dummy.position.set(-voxel.y, voxel.z, voxel.x);
      dummy.scale.setScalar(Math.max(sizeNow, 0.12) * 0.9);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      if (
        voxel.r != null &&
        voxel.g != null &&
        voxel.b != null
      ) {
        color.setRGB(voxel.r, voxel.g, voxel.b);
      } else {
        color.set(CLASS_COLOR[voxel.cls]);
      }
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

    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, MAX_INSTANCES]}
      frustumCulled={false}
      onClick={(event) => {
        event.stopPropagation();
        if (event.instanceId == null) return;
        onSelect(visibleRef.current[event.instanceId] ?? null);
      }}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial toneMapped={false} vertexColors />
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
      target={[0, 1.05, 16]}
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
          camera={{ position: [0, 1.55, 1.85], fov: 42 }}
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

          <mesh position={[0, 0.7, 0]} castShadow={false}>
            <boxGeometry args={[1.85, 1.4, 4.6]} />
            <meshBasicMaterial color="#1b1f24" />
          </mesh>

          <Grid
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
