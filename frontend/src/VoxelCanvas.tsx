import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';

interface VoxelData {
  x: number;
  y: number;
  z: number;
  prob: number;
}

interface VoxelCanvasProps {
  voxels: VoxelData[];
  maxVoxels?: number;
}

const MAX_INSTANCES = 16384;
const dummy = new THREE.Object3D();
const color = new THREE.Color();

const VoxelInstances: React.FC<{ voxels: VoxelData[] }> = ({ voxels }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null!);

  useEffect(() => {
    if (!meshRef.current) return;

    const count = Math.min(voxels.length, MAX_INSTANCES);
    meshRef.current.count = count;

    for (let i = 0; i < count; i++) {
      const v = voxels[i];
      
      // Position voxel in 3D space
      dummy.position.set(v.x * 0.5, v.z * 0.5, v.y * 0.5);
      dummy.scale.set(0.45, 0.45, 0.45);
      dummy.updateMatrix();

      meshRef.current.setMatrixAt(i, dummy.matrix);

      // Dynamic color mapping based on probability heatmap (Blue = low, Red = high)
      color.setHSL((1.0 - v.prob) * 0.66, 1.0, 0.5);
      meshRef.current.setColorAt(i, color);
    }

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [voxels]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, MAX_INSTANCES]}
      castShadow
      receiveShadow
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial roughness={0.3} metalness={0.2} />
    </instancedMesh>
  );
};

export const VoxelCanvas: React.FC<VoxelCanvasProps> = ({ voxels }) => {
  return (
    <div style={{ width: '100%', height: '100vh', background: '#0d1117' }}>
      <Canvas camera={{ position: [20, 15, 20], fov: 50 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 20, 15]} intensity={1.2} castShadow />
        <pointLight position={[-10, -10, -10]} intensity={0.4} />

        <VoxelInstances voxels={voxels} />

        {/* Spatial reference ground grid */}
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
  );
};