import React, { useRef, useMemo, type MutableRefObject } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import type { SemanticClass, SelectedVoxel, VoxelData } from './types';

interface VoxelCanvasProps {
  voxelsRef: MutableRefObject<VoxelData[]>;
  voxelSize: number;
  layers: Record<SemanticClass, boolean>;
  selected: SelectedVoxel | null;
  onSelect: (voxel: VoxelData | null) => void;
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
  spacing: number;
  layers: Record<SemanticClass, boolean>;
  selected: VoxelData | null;
  onSelect: (voxel: VoxelData | null) => void;
}> = ({ voxelsRef, spacing, layers, selected, onSelect }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const visibleRef = useRef<VoxelData[]>([]);
  const layersRef = useRef(layers);
  const spacingRef = useRef(spacing);
  const selectedRef = useRef(selected);
  const lastVoxelsRef = useRef<VoxelData[] | null>(null);
  layersRef.current = layers;
  spacingRef.current = spacing;
  selectedRef.current = selected;

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    if (voxelsRef.current === lastVoxelsRef.current) return;
    lastVoxelsRef.current = voxelsRef.current;

    const layersNow = layersRef.current;
    const spacingNow = spacingRef.current;
    const selectedNow = selectedRef.current;
    const visible = voxelsRef.current.filter((voxel) => layersNow[voxel.cls]);
    visibleRef.current = visible;

    const count = Math.min(visible.length, MAX_INSTANCES);
    mesh.count = count;

    for (let i = 0; i < count; i++) {
      const voxel = visible[i];
      dummy.position.set(voxel.x * spacingNow, voxel.z * spacingNow, voxel.y * spacingNow);
      dummy.scale.setScalar(spacingNow * 0.9);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

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

    mesh.instanceMatrix.needsUpdate = true;
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
        if (event.instanceId == null) return;
        onSelect(visibleRef.current[event.instanceId] ?? null);
      }}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
};

export const VoxelCanvas: React.FC<VoxelCanvasProps> = ({
  voxelsRef,
  voxelSize,
  layers,
  selected,
  onSelect,
}) => {
  const spacing = voxelSize * 4;
  const layerKey = useMemo(
    () => `${layers.driveable}-${layers.vehicle}-${layers.pedestrian}`,
    [layers],
  );

  return (
    <section className="pane">
      <div className="pane-header">
        <h1 className="pane-title">
          Vision Prediction
          <span className="sub">3D Voxel Grid Scene</span>
        </h1>
      </div>
      <div className="voxel-stage">
        <Canvas
          camera={{ position: [18, 14, 18], fov: 50 }}
          onPointerMissed={() => onSelect(null)}
        >
          <color attach="background" args={['#0d1117']} />
          <ambientLight intensity={0.8} />

          <VoxelInstances
            key={layerKey}
            voxelsRef={voxelsRef}
            spacing={spacing}
            layers={layers}
            selected={selected?.voxel ?? null}
            onSelect={onSelect}
          />

          <Grid
            position={[0, -0.1, 0]}
            args={[40, 40]}
            cellSize={1}
            cellThickness={1}
            cellColor="#30363d"
            sectionSize={5}
            sectionThickness={1.5}
            sectionColor="#58a6ff"
            fadeDistance={50}
          />

          <OrbitControls makeDefault enableDamping dampingFactor={0.05} />
        </Canvas>
      </div>
    </section>
  );
};
