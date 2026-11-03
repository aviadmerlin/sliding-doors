import { interpolate, useCurrentFrame } from "remotion";
import { theme } from "../theme";
import { Wordmark } from "../Wordmark";

/** The literal "sliding doors" reveal: two panels meeting at center slide
 * apart to reveal the wordmark. */
export const DoorSplit: React.FC<{ slideFrames: number }> = ({ slideFrames }) => {
  const frame = useCurrentFrame();
  const progress = interpolate(frame, [0, slideFrames], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const slide = interpolate(progress, [0, 1], [0, 1020], {
    extrapolateRight: "clamp",
  });
  const wordmarkScale = interpolate(progress, [0, 1], [0.92, 1], {
    extrapolateRight: "clamp",
  });
  const wordmarkOpacity = interpolate(progress, [0, 0.4], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          opacity: wordmarkOpacity,
          transform: `scale(${wordmarkScale})`,
        }}
      >
        <Wordmark />
      </div>
      <div
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          left: 0,
          width: 960,
          background: theme.bg,
          borderRight: `2px solid ${theme.accent}66`,
          transform: `translateX(${-slide}px)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          top: 0,
          bottom: 0,
          right: 0,
          width: 960,
          background: theme.bg,
          borderLeft: `2px solid ${theme.accent}66`,
          transform: `translateX(${slide}px)`,
        }}
      />
    </div>
  );
};
