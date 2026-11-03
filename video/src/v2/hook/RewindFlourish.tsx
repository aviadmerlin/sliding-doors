import { interpolate, useCurrentFrame } from "remotion";
import { theme } from "../theme";

/** A quick "time reversing" visual beat: a radial flash, a sweep of thin
 * vertical scrub ticks moving right-to-left, and a rewind glyph -- reads
 * as "rewind" without needing a literal reverse-played video clip. */
export const RewindFlourish: React.FC<{ durationInFrames: number }> = ({ durationInFrames }) => {
  const frame = useCurrentFrame();
  const progress = frame / durationInFrames;

  const flashOpacity = interpolate(frame, [0, 4, 14], [0, 0.5, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const ticks = Array.from({ length: 7 }, (_, i) => {
    const phase = (progress * 1.6 + i * 0.14) % 1;
    return {
      x: interpolate(phase, [0, 1], [1920 + 100, -100]),
      o: interpolate(phase, [0, 0.1, 0.8, 1], [0, 0.5, 0.5, 0]),
    };
  });

  const glyphOpacity = interpolate(frame, [2, 8, durationInFrames - 8, durationInFrames - 2], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `radial-gradient(circle at 50% 50%, ${theme.accent}55 0%, transparent 60%)`,
          opacity: flashOpacity,
        }}
      />
      {ticks.map((t, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            left: t.x,
            top: 0,
            width: 3,
            height: 1080,
            background: theme.accent,
            opacity: t.o,
          }}
        />
      ))}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          opacity: glyphOpacity,
        }}
      >
        <svg width={120} height={80} viewBox="0 0 120 80">
          <polygon points="60,10 10,40 60,70" fill={theme.text} opacity={0.9} />
          <polygon points="110,10 60,40 110,70" fill={theme.text} opacity={0.9} />
        </svg>
      </div>
    </div>
  );
};
