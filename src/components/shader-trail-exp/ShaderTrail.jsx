import { useMemo, useRef } from "react";
import * as THREE from "three/webgpu";
import { useFrame } from "@react-three/fiber";

import WebGPUScene from "./WebGPUScene";
import WebGPUSketch from "./WebGPUSketch";

import { useGridTrailTexture } from "../../lib/tsl/interactivity/use-grid-trail-texture";
import { useBayerTexture } from "../../lib/tsl/shader-trail/bayer-texture";
import { useTextCanvasTexture } from "../../lib/tsl/shader-trail/text-texture";

import {
    buildColorNode,
    createColorNodeUniforms,
} from "../../lib/tsl/shader-trail/build-color-node";


function hexToRgb(hex) {
    const v = hex.startsWith("#") ? hex.slice(1) : hex;

    return [
        Number.parseInt(v.slice(0, 2), 16) / 255,
        Number.parseInt(v.slice(2, 4), 16) / 255,
        Number.parseInt(v.slice(4, 6), 16) / 255,
    ];
}

function applyHex(u, hex) {
    const [r, g, b] = hexToRgb(hex);

    u.value.set(r, g, b);
}

function Sketch() {

    const introT = useRef(0);
    const introPointer = useRef(null);

    useFrame((_, delta) => {
        const duration = 1.4;

        if (introT.current >= 1) {
            introPointer.current = null;
            uniforms.introReveal.value = 1;
            return;
        }

        introT.current = Math.min(
            1,
            introT.current + delta / duration
        );

        const p = introT.current;

        // Smoothstep reveal
        uniforms.introReveal.value =
            p * p * (3 - 2 * p);

        // easeOutCubic cursor movement
        const t = 1 - (1 - p) ** 3;

        // Original direction: 15 degrees
        const rad = (15 * Math.PI) / 180;

        const dirX = Math.cos(rad);
        const dirY = Math.sin(rad);

        const span = Math.abs(dirX) + Math.abs(dirY);

        const wave = 0.2;

        const swing =
            Math.sin(t * Math.PI * 2) *
            wave *
            0.5;

        const uvX =
            0.5 +
            dirX * (t - 0.5) * span -
            dirY * swing;

        const uvY =
            0.5 +
            dirY * (t - 0.5) * span +
            dirX * swing;

        introPointer.current = {
            x: uvX * 2,
            y: (1 - uvY) * 2,
        };
    });

    const trailTexture = useGridTrailTexture({

        pointerOverrideRef: introPointer,

        grid: 202,
        radius: 0.05,
        strength: 0.7,
        decay: 0.97,
        influenceGain: 0.3,
        influenceGamma: 0.9,

        brushShape: "crossSquareCorner",
        filter: "nearest",

        brushRepeat: 6,
        brushThickness: 0.44,
        brushFalloff: 2.6,
        brushScale: 2,

        patternSize: 0.25,
        patternThickness: 0.1,
        patternWidth: 0.92,
    });

    const bayerTexture = useBayerTexture();

    const uniforms = useMemo(() => {
        const u = createColorNodeUniforms();

        u.filterToText.value = 1;

        u.inkEnabled.value = 1;
        u.warpStrength.value = 0.5;

        u.thresholdEnabled.value = 0;
        u.threshold.value = 0.91;
        u.softness.value = 0.36;

        u.blurEnabled.value = 0;
        u.blurStrength.value = 0.04;
        u.blurAngle.value = 0;
        u.blurSamples.value = 12;

        u.ditherEnabled.value = 0;
        u.ditherOpacity.value = 0.36;
        u.ditherPixelSize.value = 1;
        u.ditherSpread.value = 0.4;
        u.ditherLevels.value = 3;

        u.vignetteEnabled.value = 1;
        u.vignetteAmount.value = 1.5;

        return u;
    }, []);

    uniforms.trailEnabled.value = 1;
    uniforms.trailIntensity.value = 4.5;
    uniforms.trailBlurRadius.value = 0.02;

    applyHex(uniforms.trailColorLow, "#eb0000");
    applyHex(uniforms.trailColorMid, "#ffd70f");
    applyHex(uniforms.trailColorHigh, "#d8a004");

    uniforms.inkNoiseAmount.value = 1;
    uniforms.noiseScale.value = 3;
    uniforms.noiseSpeed.value = 0;

    uniforms.thresholdNoise.value = 0.46;

    uniforms.grainEnabled.value = 0;
    uniforms.grainIntensity.value = 0.08;
    uniforms.grainScale.value = 8;

    applyHex(uniforms.coreColor, "#282828");
    applyHex(uniforms.midColor, "#282828");
    applyHex(uniforms.edgeColor, "#282828");
    applyHex(uniforms.bgColor, "#f5f1e8");

    const sourceTexture = useTextCanvasTexture({
        text: "TRACE",
    });

    const colorNode = useMemo(() => {
        if (!trailTexture || !sourceTexture) return null;

        return buildColorNode(
            trailTexture,
            sourceTexture,
            bayerTexture,
            uniforms
        );
    }, [
        trailTexture,
        sourceTexture,
        bayerTexture,
        uniforms,
    ]);

    if (!colorNode) return null;

    return <WebGPUSketch colorNode={colorNode} />;
}

export default function ShaderTrail() {
    return (
        <div
            style={{
                width: "100%",
                height: "100svh",
                overflow: "hidden",
            }}
        >
            <WebGPUScene>
                <Sketch />
            </WebGPUScene>
        </div>
    );
}