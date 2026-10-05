"use client";

import {
  clamp,
  cos,
  Fn,
  float,
  floor,
  fract,
  Loop,
  mix,
  mod,
  pow,
  screenSize,
  sin,
  smoothstep,
  texture,
  time,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from "three/tsl";

import * as THREE from "three/webgpu";

import { simplexNoise3d } from "../noise/simplex-noise-3d";
import { bloom } from "../utils/function/bloom";

/**
 * Trail-aware text shader pipeline (TSL).
 *
 * Inputs:
 *   - trailTex : THREE.DataTexture
 *                RGBA float, from useGridTrailTexture
 *                R = inverted dx flow, G = dy flow, B = speed
 *
 *   - sourceTex: THREE.CanvasTexture or DataTexture
 *                Base content sampled at the warped UV.
 *                RGB is the visible base when `inkEnabled = 0`;
 *                luminance is the text mask.
 *
 *   - bayerTex : THREE.DataTexture
 *                4x4 Bayer ordered-dither pattern.
 *
 *   - u        : ColorNodeUniforms
 *
 * Pipeline:
 *   trail
 *   → UV warp by trail flow
 *   → sample source RGB
 *   → luminance = textMask
 *   → directional noise smear
 *   → smoothstep threshold + hash jitter
 *   → 3-stop ink gradient
 *   → grain
 *   → Bayer 4x4 ordered dither
 *   → 3-stop coloured trail by speed
 *   → radial vignette
 *
 * `filterToText` (0/1):
 * When 1, post-composition effects are masked by source luminance.
 */

const HEX = (h) => {
  const v = h.startsWith("#")
    ? h.slice(1)
    : h;

  const r =
    Number.parseInt(v.slice(0, 2), 16) / 255;

  const g =
    Number.parseInt(v.slice(2, 4), 16) / 255;

  const b =
    Number.parseInt(v.slice(4, 6), 16) / 255;

  return new THREE.Vector3(r, g, b);
};

export function createColorNodeUniforms() {
  return {
    warpStrength: uniform(0.45),

    noiseScale: uniform(1.2),

    noiseSpeed: uniform(0.36),

    threshold: uniform(0.54),

    softness: uniform(0.01),

    thresholdNoise: uniform(0.08),

    inkNoiseAmount: uniform(1),

    blurStrength: uniform(0.044),

    blurAngle: uniform(0),

    blurSamples: uniform(12),

    grainIntensity: uniform(0.3),

    grainScale: uniform(1.0),

    vignetteAmount: uniform(0.45),

    textAspect: uniform(1.0),

    textAspectY: uniform(1.0),

    textAspectB: uniform(1.0),

    textAspectBY: uniform(1.0),

    imageMix: uniform(0),

    sourceScale: uniform(1.0),

    introReveal: uniform(1),

    introBlur: uniform(0.02),

    ditherOpacity: uniform(0.17),

    ditherPixelSize: uniform(2),

    ditherSpread: uniform(0.44),

    ditherLevels: uniform(3),

    trailBlurRadius: uniform(0.01),

    trailIntensity: uniform(0.8),

    trailBloomEnabled: uniform(1),

    trailBloomEdge: uniform(0.05),

    trailBloomExponent: uniform(1.4),

    trailColorLow: uniform(
      HEX("#1a4d70")
    ),

    trailColorMid: uniform(
      HEX("#c97a3a")
    ),

    trailColorHigh: uniform(
      HEX("#fff0d4")
    ),

    inkEnabled: uniform(1),

    thresholdEnabled: uniform(1),

    blurEnabled: uniform(1),

    grainEnabled: uniform(1),

    ditherEnabled: uniform(1),

    vignetteEnabled: uniform(1),

    trailEnabled: uniform(1),

    filterToText: uniform(0),

    coreColor: uniform(
      HEX("#BDBDB9")
    ),

    midColor: uniform(
      HEX("#676351")
    ),

    edgeColor: uniform(
      HEX("#A3A3A3")
    ),

    bgColor: uniform(
      HEX("#000000")
    ),
  };
};

export function buildColorNode(
  trailTex,
  sourceTex,
  bayerTex,
  u,
  sourceTexB = null,
  mixNode = null,
  trailUvNode = null,
  unwarpedB = false
) {
  return Fn(() => {
    const _uv = uv();

    const trailUv =
      trailUvNode ?? _uv;

    // ─────────────────────────────────────
    // 9-tap Gaussian blur of the trail
    // ─────────────────────────────────────

    const r =
      u.trailBlurRadius;

    const t00 = texture(
      trailTex,
      trailUv.add(
        vec2(
          r.negate(),
          r.negate()
        )
      )
    );

    const t10 = texture(
      trailTex,
      trailUv.add(
        vec2(0, r.negate())
      )
    );

    const t20 = texture(
      trailTex,
      trailUv.add(
        vec2(r, r.negate())
      )
    );

    const t01 = texture(
      trailTex,
      trailUv.add(
        vec2(r.negate(), 0)
      )
    );

    const t11 = texture(
      trailTex,
      trailUv
    );

    const t21 = texture(
      trailTex,
      trailUv.add(
        vec2(r, 0)
      )
    );

    const t02 = texture(
      trailTex,
      trailUv.add(
        vec2(r.negate(), r)
      )
    );

    const t12 = texture(
      trailTex,
      trailUv.add(
        vec2(0, r)
      )
    );

    const t22 = texture(
      trailTex,
      trailUv.add(
        vec2(r, r)
      )
    );

    const trailBlurred =
      t00
        .add(t20)
        .add(t02)
        .add(t22)
        .add(t10.mul(2))
        .add(t01.mul(2))
        .add(t21.mul(2))
        .add(t12.mul(2))
        .add(t11.mul(4))
        .div(16);

    const flow = vec2(
      trailBlurred.r,
      trailBlurred.g
    );

    const speed = clamp(
      trailBlurred.b,
      0,
      1
    );

    // ─────────────────────────────────────
    // Warp UVs by trail flow
    // ─────────────────────────────────────

    const warpedUv =
      _uv.add(
        flow.mul(
          u.warpStrength
        )
      );

    // ─────────────────────────────────────
    // Sample source texture
    // ─────────────────────────────────────

    const centered =
      warpedUv
        .sub(0.5)
        .mul(u.sourceScale);

    const textUv =
      vec2(
        centered.x.mul(
          u.textAspect
        ),
        centered.y.mul(
          u.textAspectY
        )
      ).add(0.5);

    const introBlurRadius =
      u.introBlur
        .mul(
          float(1).sub(
            u.introReveal
          )
        )
        .max(0);

    const tapAt = (ox, oy) =>
      texture(
        sourceTex,
        textUv.add(
          vec2(ox, oy).mul(
            introBlurRadius
          )
        )
      ).rgb;

    const sourceColorA =
      tapAt(0, 0)
        .add(tapAt(1, 1))
        .add(tapAt(-1, 1))
        .add(tapAt(1, -1))
        .add(tapAt(-1, -1))
        .div(5);

    const sourceColor =
      sourceTexB
        ? (() => {
            const centeredB =
              unwarpedB
                ? _uv
                    .sub(0.5)
                    .mul(
                      u.sourceScale
                    )
                : centered;

            const textUvB =
              vec2(
                centeredB.x.mul(
                  u.textAspectB
                ),
                centeredB.y.mul(
                  u.textAspectBY
                )
              ).add(0.5);

            const sourceSampleB =
              texture(
                sourceTexB,
                textUvB
              );

            const mixFactor =
              clamp(
                mixNode ??
                  u.imageMix,
                0,
                1
              );

            return mix(
              sourceColorA,
              sourceSampleB.rgb,
              mixFactor
            );
          })()
        : sourceColorA;

    const textMaskRaw =
      sourceColor.dot(
        vec3(
          0.299,
          0.587,
          0.114
        )
      );

    // ─────────────────────────────────────
    // Directional smear
    // ─────────────────────────────────────

    const dirX =
      cos(u.blurAngle);

    const dirY =
      sin(u.blurAngle);

    const accum =
      vec3(0, 0, 0).toVar();

    const totalWeight =
      float(0).toVar();

    const samples =
      u.blurSamples.toInt();

    Loop(
      {
        start: 0,
        end: samples,
        type: "int",
      },
      ({ i }) => {
        const t =
          float(i).div(
            float(samples)
              .sub(1)
              .max(1)
          );

        const offset =
          t
            .sub(0.5)
            .mul(
              u.blurStrength
            );

        const offUv =
          warpedUv.add(
            vec2(
              dirX.mul(offset),
              dirY.mul(offset)
            )
          );

        const n =
          simplexNoise3d(
            vec4(
              offUv.mul(
                u.noiseScale
              ),
              time.mul(
                u.noiseSpeed
              ),
              0
            )
          );

        const w = float(1);

        accum.addAssign(
          vec3(n, n, n).mul(w)
        );

        totalWeight.addAssign(w);
      }
    );

    const noiseBlurredOn =
      accum
        .div(
          totalWeight.max(1)
        )
        .x
        .add(1)
        .mul(0.5);

    const noiseCentre =
      simplexNoise3d(
        vec4(
          warpedUv.mul(
            u.noiseScale
          ),
          time.mul(
            u.noiseSpeed
          ),
          0
        )
      )
        .add(1)
        .mul(0.5);

    const noiseBlurred =
      mix(
        noiseCentre,
        noiseBlurredOn,
        u.blurEnabled
      );

    // ─────────────────────────────────────
    // Intro reveal
    // ─────────────────────────────────────

    const introMask =
      clamp(
        u.introReveal,
        0,
        1
      );

    const textMask =
      textMaskRaw.mul(
        introMask
      );

    // ─────────────────────────────────────
    // Threshold + jitter
    // ─────────────────────────────────────

    const jitterSeed =
      _uv.add(
        vec2(
          time.mul(0.013),
          time.mul(0.017)
        )
      );

    const jitter =
      fract(
        jitterSeed
          .dot(
            vec2(
              91.534,
              47.871
            )
          )
          .sin()
          .mul(43758.5453)
      )
        .sub(0.5)
        .mul(
          u.thresholdNoise
        );

    const tField =
      noiseBlurred.add(
        jitter
      );

    const lo =
      u.threshold.sub(
        u.softness
      );

    const hi =
      u.threshold.add(
        u.softness
      );

    const inkThresholded =
      smoothstep(
        lo,
        hi,
        tField
      );

    const inkRaw =
      clamp(
        tField,
        0,
        1
      );

    const inkShaded =
      mix(
        inkRaw,
        inkThresholded,
        u.thresholdEnabled
      );

    const ink =
      mix(
        float(1),
        inkShaded,
        u.inkNoiseAmount
      );

    // ─────────────────────────────────────
    // 3-stop ink gradient
    // ─────────────────────────────────────

    const tFade =
      mix(
        float(1),
        clamp(
          tField,
          0,
          1
        ),
        u.inkNoiseAmount
      );

    const stop1 =
      smoothstep(
        0.0,
        0.5,
        tFade
      );

    const stop2 =
      smoothstep(
        0.5,
        1.0,
        tFade
      );

    const inkBody =
      mix(
        mix(
          u.edgeColor,
          u.midColor,
          stop1
        ),
        u.coreColor,
        stop2
      );

    const inkOverlay =
      mix(
        u.bgColor,
        inkBody.mul(ink),
        textMask
      );

    const composed =
      mix(
        sourceColor,
        inkOverlay,
        u.inkEnabled
      );

    // ─────────────────────────────────────
    // Filter-to-text mask
    // ─────────────────────────────────────

    const fxMask =
      mix(
        float(1),
        textMask,
        u.filterToText
      );

    // ─────────────────────────────────────
    // Grain
    // ─────────────────────────────────────

    const grainUv =
      _uv
        .mul(u.grainScale)
        .add(
          vec2(
            time,
            time.mul(0.7)
          )
        );

    const grain =
      fract(
        grainUv
          .dot(
            vec2(
              12.9898,
              78.233
            )
          )
          .sin()
          .mul(43758.5453)
      );

    const grainShift =
      grain
        .sub(0.5)
        .mul(
          u.grainIntensity
        )
        .mul(
          u.grainEnabled
        )
        .mul(fxMask);

    const withGrain =
      composed.add(
        vec3(
          grainShift,
          grainShift,
          grainShift
        )
      );

    // ─────────────────────────────────────
    // Bayer 4x4 ordered dither
    // ─────────────────────────────────────

    const pixelCoord =
      uv()
        .mul(screenSize)
        .div(
          u.ditherPixelSize.max(
            1
          )
        );

    const cell =
      mod(
        floor(pixelCoord),
        4
      );

    const bayerUv =
      cell
        .add(0.5)
        .div(4);

    const bayerVal =
      texture(
        bayerTex,
        bayerUv
      ).r;

    const bias =
      bayerVal
        .sub(0.5)
        .mul(
          u.ditherSpread
        );

    const biased =
      withGrain.add(
        vec3(
          bias,
          bias,
          bias
        )
      );

    const lvl =
      u.ditherLevels.max(2);

    const lvlMinusOne =
      lvl.sub(1);

    const quantized =
      floor(
        biased
          .mul(lvlMinusOne)
          .add(0.5)
      ).div(
        lvlMinusOne
      );

    const dithered =
      mix(
        withGrain,
        quantized,
        u.ditherOpacity
          .mul(
            u.ditherEnabled
          )
          .mul(fxMask)
      );

    // ─────────────────────────────────────
    // 3-stop coloured trail
    // ─────────────────────────────────────

    const tSpeed =
      clamp(
        speed,
        0,
        1
      );

    const tLowMid =
      smoothstep(
        0,
        0.5,
        tSpeed
      );

    const tMidHigh =
      smoothstep(
        0.5,
        1,
        tSpeed
      );

    const trailColor =
      mix(
        mix(
          u.trailColorLow,
          u.trailColorMid,
          tLowMid
        ),
        u.trailColorHigh,
        tMidHigh
      );

    // ─────────────────────────────────────
    // Bloom
    // ─────────────────────────────────────

    const bloomPattern =
      float(1)
        .sub(tSpeed)
        .max(0.001)
        .toVar();

    const bloomGlow =
      clamp(
        bloom(
          bloomPattern,
          u.trailBloomEdge,
          u.trailBloomExponent
        ),
        0,
        4
      );

    const trailBoost =
      mix(
        float(1),
        float(1).add(
          bloomGlow
        ),
        u.trailBloomEnabled
      );

    const trailContribution =
      trailColor
        .mul(tSpeed)
        .mul(trailBoost)
        .mul(
          u.trailIntensity
        )
        .mul(
          u.trailEnabled
        )
        .mul(fxMask);

    // Chain trail on top of the fully
    // processed colour.
    const lifted =
      dithered.add(
        trailContribution
      );

    // ─────────────────────────────────────
    // Radial vignette
    // ─────────────────────────────────────

    const cv =
      _uv.sub(0.5);

    const dist =
      cv
        .dot(cv)
        .mul(2);

    const vig =
      clamp(
        float(1).sub(
          dist.mul(
            u.vignetteAmount
          )
        ),
        0,
        1
      );

    const vigBaseOrOne =
      mix(
        float(1),
        pow(
          vig,
          float(1.4)
        ),
        u.vignetteEnabled
      );

    const vigFactor =
      mix(
        float(1),
        vigBaseOrOne,
        fxMask
      );

    return lifted;
  })();
}