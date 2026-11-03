import { Audio } from "@remotion/media";
import { AbsoluteFill, Sequence, staticFile } from "remotion";
import { Background } from "../Background";
import { KaraokeCaptions } from "../KaraokeCaptions";
import hookCaptions from "../../../public/audio-v2/01-hook.captions.json";
import { DoorSplit } from "./DoorSplit";
import { FreezeShot } from "./FreezeShot";
import { RewindFlourish } from "./RewindFlourish";

export const HOOK_SCENE_DURATION = 210; // 7.0s @ 30fps

export const HookScene: React.FC = () => {
  return (
    <AbsoluteFill>
      <Background />

      <Sequence  durationInFrames={100}>
        <FreezeShot />
      </Sequence>

      <Sequence from={72} durationInFrames={26}>
        <RewindFlourish durationInFrames={26} />
      </Sequence>

      <Sequence from={100} durationInFrames={HOOK_SCENE_DURATION - 100}>
        <DoorSplit slideFrames={35} />
      </Sequence>

      <Audio src={staticFile("audio-v2/01-hook.mp3")} />
      <KaraokeCaptions captions={hookCaptions} />
    </AbsoluteFill>
  );
};
