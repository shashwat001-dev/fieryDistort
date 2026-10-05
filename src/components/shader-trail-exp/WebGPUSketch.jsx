import { useMemo } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three/webgpu";

export default function WebGPUSketch({ colorNode }) {
  const { size } = useThree();

  const material = useMemo(() => {
    const value = new THREE.MeshBasicNodeMaterial({
      transparent: true,
    });

    value.colorNode = colorNode;

    return value;
  }, [colorNode]);

  return (
    <mesh
      material={material}
      scale={[size.width / size.height, 1, 1]}
    >
      <planeGeometry args={[2, 2]} />
    </mesh>
  );
}