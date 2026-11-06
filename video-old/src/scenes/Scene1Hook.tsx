import { AbsoluteFill } from "remotion";
import { Clip } from "../Clip";
import { VoiceOver } from "../VoiceOver";

export const SCENE_1_DURATION_SEC = 10.0;

export const Scene1Hook: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: "#05070c" }}>
      <Clip
        src="recordings/desktop.webm"
        fromSec={0}
        trimBeforeSec={0}
        sourceDurationSec={SCENE_1_DURATION_SEC}
      />
      <VoiceOver file="01-hook.mp3" />
    </AbsoluteFill>
  );
};
