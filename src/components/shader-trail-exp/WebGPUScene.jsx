import { useState } from "react";
import * as THREE from "three/webgpu";
import { Canvas } from "@react-three/fiber";

export default function WebGPUScene({ children }) {
  const [ready, setReady] = useState(false);

  return (
    <Canvas
      style={{
        width: "100%",
        height: "100%",
        display: "block",
      }}
      frameloop={ready ? "always" : "never"}
      dpr={[1, 1]}
      camera={{
        position: [0, 0, 1],
      }}
      gl={async (canvasProps) => {
        const renderer = new THREE.WebGPURenderer(canvasProps);

        await renderer.init();

        setReady(true);

        return renderer;
      }}
    >
      {ready && children}
    </Canvas>
  );
}