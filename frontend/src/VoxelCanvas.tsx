import React, { useRef, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import type { SemanticClass, SelectedVoxel, VoxelData } from './types';

interface VoxelCanvasProps {
  voxels: VoxelData[];
  voxelSize: number;
  layers: Record<SemanticClass, boolean>;
  selected: SelectedVoxel | null;
  onSelect: (voxel: VoxelData | null) => void;
}

const MAX_INSTANCES = 16384;
const dummy = new THREE.Object3D();
const color = new THREE.Color();

const CLASS_COLOR: Record<SemanticClass, string> = {
  driveable: '#7ee787',
  vehicle: '#58a6ff',
  pedestrian: '#f85149',
};

const VoxelInstances: React.FC<{
  voxels: VoxelData[];
  spacing: number;
  selected: VoxelData | null;
  onSelect: (voxel: VoxelData | null) => void;
}> = ({ voxels, spacing, selected, onSelect }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null!);

  useEffect(() => {
    if (!meshRef.current) return;

    const count = Math.min(voxels.length, MAX_INSTANCES);
    meshRef.current.count = count;

    for (let i = 0; i < count; i++) {
      const v = voxels[i];
      dummy.position.set(v.x * spacing, v.z * spacing, v.y * spacing);
      dummy.scale.setScalar(spacing * 0.9);
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);

      color.set(CLASS_COLOR[v.cls]);
      if (selected && selected.x === v.x && selected.y === v.y && selected.z === v.z) {
        color.offsetHSL(0, 0, 0.25);
      }
      meshRef.current.setColorAt(i, color);
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [voxels, spacing, selected]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, MAX_INSTANCES]}
      castShadow
      receiveShadow
      onClick={(event) => {
        event.stopPropagation();
        if (event.instanceId == null) return;
        onSelect(voxels[event.instanceId] ?? null);
      }}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial roughness={0.35} metalness={0.15} />
    </instancedMesh>
  );
};

export const VoxelCanvas: React.FC<VoxelCanvasProps> = ({
  voxels,
  voxelSize,
  layers,
  selected,
  onSelect,
}) => {
  const spacing = voxelSize * 4;
  const visible = useMemo(
    () => voxels.filter((v) => layers[v.cls]),
    [voxels, layers],
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
          <ambientLight intensity={0.6} />
          <directionalLight position={[10, 20, 15]} intensity={1.2} castShadow />
          <pointLight position={[-10, -10, -10]} intensity={0.4} />

          <VoxelInstances
            voxels={visible}
            spacing={spacing}
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
