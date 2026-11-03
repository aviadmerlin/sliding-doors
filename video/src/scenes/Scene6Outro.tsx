import { Video } from "@remotion/media";
import {
  AbsoluteFill,
  Sequence,
  interpolate,
  staticFile,
  useCurrentFrame,
} from "remotion";
import { mobileMarks } from "../marks";
import { VoiceOver } from "../VoiceOver";

const MOBILE_DURATION = 5.0;
const TITLE_DURATION = 4.5;

export const SCENE_6_DURATION_SEC = MOBILE_DURATION + TITLE_DURATION;

const MobileFrame: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 12], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <AbsoluteFill
      style={{
        background: "#05070c",
        alignItems: "center",
        justifyContent: "center",
        opacity,
      }}
    >
      <div
        style={{
          width: 470,
          height: 1016,
          borderRadius: 48,
          border: "10px solid #1b212c",
          background: "#000",
          overflow: "hidden",
          boxShadow: "0 40px 120px rgba(0,0,0,0.55)",
          position: "relative",
        }}
      >
        <Video
          src={staticFile("recordings/mobile.webm")}
          trimBefore={Math.round(mobileMarks.heroVisible * 30)}
          durationInFrames={Math.round(MOBILE_DURATION * 30)}
          muted
          objectFit="cover"
          style={{ width: "100%", height: "100%" }}
        />
      </div>
    </AbsoluteFill>
  );
};

const TitleCard: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const rise = interpolate(frame, [0, 20], [24, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <AbsoluteFill
      style={{
        background: "#05070c",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        opacity,
        transform: `translateY(${rise}px)`,
      }}
    >
      <div
        style={{
          fontFamily:
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          fontSize: 140,
          fontWeight: 800,
          color: "#f2f4f8",
          letterSpacing: -2,
        }}
      >
        Sliding Doors
      </div>
      <div
        style={{
          marginTop: 28,
          fontFamily:
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          fontSize: 48,
          fontWeight: 500,
          color: "#8fe3a8",
        }}
      >
        Every match, every moment, every what if.
      </div>
    </AbsoluteFill>
  );
};

export const Scene6Outro: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: "#05070c" }}>
      <Sequence  durationInFrames={Math.round(MOBILE_DURATION * 30)}>
        <MobileFrame />
      </Sequence>
      <Sequence
        from={Math.round(MOBILE_DURATION * 30)}
        durationInFrames={Math.round(TITLE_DURATION * 30)}
      >
        <TitleCard />
      </Sequence>
      <VoiceOver file="06-outro.mp3" />
    </AbsoluteFill>
  );
};
