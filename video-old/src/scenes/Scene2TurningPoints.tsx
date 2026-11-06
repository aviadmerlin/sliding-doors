import { AbsoluteFill } from "remotion";
import { Caption } from "../Caption";
import { Clip } from "../Clip";
import { VoiceOver } from "../VoiceOver";
import { SCENE_1_DURATION_SEC } from "./Scene1Hook";

export const SCENE_2_DURATION_SEC = 17.5;

// Picks up exactly where Scene 1 left off in the source recording -- a
// continuous, uncut pass through the scoreboard-to-chart scroll.
const SOURCE_START_SEC = SCENE_1_DURATION_SEC;

export const Scene2TurningPoints: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: "#05070c" }}>
      <Clip
        src="recordings/desktop.webm"
        fromSec={0}
        trimBeforeSec={SOURCE_START_SEC}
        sourceDurationSec={SCENE_2_DURATION_SEC}
      />
      <Caption fromFrame={90}>
        Ranked by how much each moment really changed the outcome
      </Caption>
      <VoiceOver file="02-turning-points.mp3" />
    </AbsoluteFill>
  );
};
