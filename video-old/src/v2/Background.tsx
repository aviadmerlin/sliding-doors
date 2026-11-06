import { AbsoluteFill, useCurrentFrame } from "remotion";
import { theme } from "./theme";

/**
 * The animated dark backdrop used behind every non-captured-footage
 * moment -- never flat, never white. A slow gradient + a faint drifting
 * grid + two soft accent-tinted glows. See DESIGN.md "Palette".
 */
export const Background: React.FC = () => {
  const frame = useCurrentFrame();
  const drift = frame * 0.15;

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(120% 90% at 30% 20%, ${theme.bgSidebar} 0%, ${theme.bg} 55%, #060910 100%)`,
      }}
    >
      {/* faint drifting grid */}
      <AbsoluteFill
        style={{
          backgroundImage: `linear-gradient(${theme.border}22 1px, transparent 1px), linear-gradient(90deg, ${theme.border}22 1px, transparent 1px)`,
          backgroundSize: "64px 64px",
          backgroundPosition: `${drift}px ${drift * 0.6}px`,
          opacity: 0.35,
        }}
      />
      {/* soft accent glows */}
      <div
        style={{
          position: "absolute",
          width: 900,
          height: 900,
          borderRadius: "50%",
          left: -250 + Math.sin(frame / 140) * 40,
          top: -300 + Math.cos(frame / 160) * 30,
          background: `radial-gradient(circle, ${theme.accent}22 0%, transparent 70%)`,
          filter: "blur(10px)",
        }}
      />
      <div
        style={{
          position: "absolute",
          width: 700,
          height: 700,
          borderRadius: "50%",
          right: -200 + Math.cos(frame / 150) * 35,
          bottom: -250 + Math.sin(frame / 130) * 25,
          background: `radial-gradient(circle, ${theme.purple}18 0%, transparent 70%)`,
          filter: "blur(10px)",
        }}
      />
    </AbsoluteFill>
  );
};
