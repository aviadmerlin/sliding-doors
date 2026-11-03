import { Video } from "@remotion/media";
import type { CSSProperties } from "react";
import { Sequence, staticFile, useVideoConfig } from "remotion";

const fullBleed: CSSProperties = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
};

/**
 * One cut from a recorded source file, placed at `fromSec` within its
 * parent scene. `trimBeforeSec`/`sourceDurationSec` address the source
 * clip's own timeline; `playbackRate` lets a cut (e.g. a computation wait)
 * play faster than real time while still being real recorded footage.
 */
export const Clip: React.FC<{
  src: string;
  fromSec: number;
  trimBeforeSec: number;
  sourceDurationSec: number;
  playbackRate?: number;
  style?: CSSProperties;
}> = ({ src, fromSec, trimBeforeSec, sourceDurationSec, playbackRate = 1, style }) => {
  const { fps } = useVideoConfig();
  const outputDurationSec = sourceDurationSec / playbackRate;
  return (
    <Sequence
      from={Math.round(fromSec * fps)}
      durationInFrames={Math.round(outputDurationSec * fps)}
      name={`${src} @${trimBeforeSec.toFixed(1)}s`}
    >
      <Video
        src={staticFile(src)}
        trimBefore={Math.round(trimBeforeSec * fps)}
        durationInFrames={Math.round(outputDurationSec * fps)}
        playbackRate={playbackRate}
        muted
        objectFit="cover"
        style={{ ...fullBleed, ...style }}
      />
    </Sequence>
  );
};
