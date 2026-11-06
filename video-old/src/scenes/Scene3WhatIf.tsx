import { AbsoluteFill } from "remotion";
import { Caption } from "../Caption";
import { Clip } from "../Clip";
import { VoiceOver } from "../VoiceOver";
import { desktopMarks } from "../marks";
import { SCENE_1_DURATION_SEC } from "./Scene1Hook";
import { SCENE_2_DURATION_SEC } from "./Scene2TurningPoints";

const SOURCE_START_SEC = SCENE_1_DURATION_SEC + SCENE_2_DURATION_SEC;

// a: lead-in (scroll settle, pick the 84' event, click Run) at real speed.
const A_DURATION = desktopMarks.runCounterfactualClicked - SOURCE_START_SEC;
// c: the real ~7s Monte Carlo wait, sped up -- still the real computation,
// just not shown in real time.
const C_SOURCE_DURATION =
  desktopMarks.counterfactualResultVisible - desktopMarks.runCounterfactualClicked;
const C_PLAYBACK_RATE = 2.333;
const C_DURATION = C_SOURCE_DURATION / C_PLAYBACK_RATE;
// b: hold on the real "9% -> 92%" result.
const B_DURATION = 5.0;

export const SCENE_3_DURATION_SEC = A_DURATION + C_DURATION + B_DURATION;

export const Scene3WhatIf: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: "#05070c" }}>
      <Clip
        src="recordings/desktop.webm"
        fromSec={0}
        trimBeforeSec={SOURCE_START_SEC}
        sourceDurationSec={A_DURATION}
      />
      <Clip
        src="recordings/desktop.webm"
        fromSec={A_DURATION}
        trimBeforeSec={desktopMarks.runCounterfactualClicked}
        sourceDurationSec={C_SOURCE_DURATION}
        playbackRate={C_PLAYBACK_RATE}
      />
      <Clip
        src="recordings/desktop.webm"
        fromSec={A_DURATION + C_DURATION}
        trimBeforeSec={desktopMarks.counterfactualResultVisible}
        sourceDurationSec={B_DURATION}
      />
      <Caption fromFrame={Math.round((A_DURATION + C_DURATION) * 30) + 6}>
        Dunmore Town win probability: 9% → 92%
      </Caption>
      <VoiceOver file="03-whatif.mp3" />
    </AbsoluteFill>
  );
};
