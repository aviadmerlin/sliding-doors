import { Composition } from "remotion";
import { FPS, SlidingDoorsDemo, TOTAL_DURATION_IN_FRAMES } from "./SlidingDoorsDemo";
import { HookScene, HOOK_SCENE_DURATION } from "./v2/hook/HookScene";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="SlidingDoorsDemo"
        component={SlidingDoorsDemo}
        durationInFrames={TOTAL_DURATION_IN_FRAMES}
        fps={FPS}
        width={1920}
        height={1080}
      />
      {/* v2 rebuild, built scene by scene -- see video/STORYBOARD.md */}
      <Composition
        id="HookScene"
        component={HookScene}
        durationInFrames={HOOK_SCENE_DURATION}
        fps={FPS}
        width={1920}
        height={1080}
      />
    </>
  );
};
