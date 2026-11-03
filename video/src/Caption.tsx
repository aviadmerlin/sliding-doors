import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";

/** Lower-third caption bar, safe-area padded, fading in and out. */
export const Caption: React.FC<{ children: string; fromFrame?: number }> = ({
  children,
  fromFrame = 0,
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const local = frame - fromFrame;
  if (local < 0) return null;

  const fadeIn = interpolate(local, [0, 12], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const fadeOut = interpolate(
    local,
    [durationInFrames - fromFrame - 12, durationInFrames - fromFrame],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const opacity = Math.min(fadeIn, fadeOut);
  const rise = interpolate(local, [0, 12], [12, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        position: "absolute",
        left: 140,
        right: 140,
        bottom: 110,
        display: "flex",
        justifyContent: "center",
        opacity,
        transform: `translateY(${rise}px)`,
      }}
    >
      <div
        style={{
          maxWidth: 1400,
          background: "rgba(8, 12, 20, 0.84)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 16,
          padding: "22px 34px",
          color: "#f2f4f8",
          fontFamily:
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          fontSize: 40,
          fontWeight: 600,
          lineHeight: 1.35,
          textAlign: "center",
          boxShadow: "0 12px 40px rgba(0,0,0,0.35)",
        }}
      >
        {children}
      </div>
    </div>
  );
};
