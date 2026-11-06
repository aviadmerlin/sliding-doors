import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { AbsoluteFill } from "remotion";
import { Scene1Hook, SCENE_1_DURATION_SEC } from "./scenes/Scene1Hook";
import { Scene2TurningPoints, SCENE_2_DURATION_SEC } from "./scenes/Scene2TurningPoints";
import { Scene3WhatIf, SCENE_3_DURATION_SEC } from "./scenes/Scene3WhatIf";
import { Scene4Agent, SCENE_4_DURATION_SEC } from "./scenes/Scene4Agent";
import { Scene5Calibration, SCENE_5_DURATION_SEC } from "./scenes/Scene5Calibration";
import { Scene6Outro, SCENE_6_DURATION_SEC } from "./scenes/Scene6Outro";

export const FPS = 30;
const TRANSITION_FRAMES = 10;

const sceneDurationsSec = [
  SCENE_1_DURATION_SEC,
  SCENE_2_DURATION_SEC,
  SCENE_3_DURATION_SEC,
  SCENE_4_DURATION_SEC,
  SCENE_5_DURATION_SEC,
  SCENE_6_DURATION_SEC,
];
const sceneFrames = sceneDurationsSec.map((s) => Math.round(s * FPS));
const transitionCount = sceneFrames.length - 1;

export const TOTAL_DURATION_IN_FRAMES =
  sceneFrames.reduce((a, b) => a + b, 0) - transitionCount * TRANSITION_FRAMES;

const Wordmark: React.FC = () => (
  <div
    style={{
      position: "absolute",
      top: 48,
      left: 56,
      display: "flex",
      alignItems: "center",
      gap: 10,
      padding: "10px 20px",
      background: "rgba(8, 12, 20, 0.6)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 999,
      color: "#f2f4f8",
      fontFamily:
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      fontSize: 24,
      fontWeight: 700,
      letterSpacing: 0.3,
      zIndex: 10,
    }}
  >
    ⚽ Sliding Doors
  </div>
);

export const SlidingDoorsDemo: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: "#05070c" }}>
      <TransitionSeries>
        <TransitionSeries.Sequence name="Hook" durationInFrames={sceneFrames[0]}>
          <Scene1Hook />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          presentation={fade()}
          timing={linearTiming({ durationInFrames: TRANSITION_FRAMES })}
        />
        <TransitionSeries.Sequence name="Turning points" durationInFrames={sceneFrames[1]}>
          <Scene2TurningPoints />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          presentation={fade()}
          timing={linearTiming({ durationInFrames: TRANSITION_FRAMES })}
        />
        <TransitionSeries.Sequence name="What if" durationInFrames={sceneFrames[2]}>
          <Scene3WhatIf />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          presentation={fade()}
          timing={linearTiming({ durationInFrames: TRANSITION_FRAMES })}
        />
        <TransitionSeries.Sequence name="Agent" durationInFrames={sceneFrames[3]}>
          <Scene4Agent />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          presentation={fade()}
          timing={linearTiming({ durationInFrames: TRANSITION_FRAMES })}
        />
        <TransitionSeries.Sequence name="Calibration" durationInFrames={sceneFrames[4]}>
          <Scene5Calibration />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          presentation={fade()}
          timing={linearTiming({ durationInFrames: TRANSITION_FRAMES })}
        />
        <TransitionSeries.Sequence name="Outro" durationInFrames={sceneFrames[5]}>
          <Scene6Outro />
        </TransitionSeries.Sequence>
      </TransitionSeries>
      <Wordmark />
    </AbsoluteFill>
  );
};
