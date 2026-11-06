import { AbsoluteFill, Img, Sequence, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Caption } from "../Caption";
import { Clip } from "../Clip";
import { VoiceOver } from "../VoiceOver";
import { desktopMarks } from "../marks";

const CALIBRATION_DURATION = 8.0;
const CLUB_PROFILE_DURATION = 6.0;

export const SCENE_5_DURATION_SEC = CALIBRATION_DURATION + CLUB_PROFILE_DURATION;

const CalibrationPan: React.FC = () => {
  const frame = useCurrentFrame();
  const durationInFrames = Math.round(CALIBRATION_DURATION * 30);
  const scale = interpolate(frame, [0, durationInFrames], [1.0, 1.12], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <AbsoluteFill style={{ background: "#0b0e14", overflow: "hidden" }}>
      <Img
        src={staticFile("calibration.png")}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "contain",
          transform: `scale(${scale})`,
        }}
      />
      <Caption fromFrame={20}>
        Under the hood: 5 seasons · 1,900 matches · validated out-of-sample
      </Caption>
    </AbsoluteFill>
  );
};

export const Scene5Calibration: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: "#05070c" }}>
      <Sequence  durationInFrames={Math.round(CALIBRATION_DURATION * 30)}>
        <CalibrationPan />
      </Sequence>
      <Clip
        src="recordings/desktop.webm"
        fromSec={CALIBRATION_DURATION}
        trimBeforeSec={desktopMarks.clubProfileVisible}
        sourceDurationSec={CLUB_PROFILE_DURATION}
      />
      <VoiceOver file="05-calibration.mp3" />
    </AbsoluteFill>
  );
};
