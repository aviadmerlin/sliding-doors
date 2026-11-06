import { interpolate, useCurrentFrame } from "remotion";
import { spaceGrotesk } from "../fonts";
import { theme } from "../theme";

const HEADLINE_WORDS = ["What", "if", "this", "shot", "went", "in?"];
const TYPE_START = 8;
const FRAMES_PER_WORD = 7;

/** A minimal, abstract frozen shot-on-goal graphic -- not photorealistic,
 * matching the app's own clean data-product aesthetic. A ball travels an
 * arc toward a goal line and freezes mid-flight with a motion trail. */
const BallAndGoal: React.FC<{ progress: number }> = ({ progress }) => {
  // progress 0 = ball at penalty spot, 1 = ball at goal line.
  const x = interpolate(progress, [0, 1], [640, 1320]);
  const y = interpolate(progress, [0, 1], [680, 420]);
  const trail = Array.from({ length: 6 }, (_, i) => {
    const p = Math.max(0, progress - i * 0.05);
    return {
      x: interpolate(p, [0, 1], [640, 1320]),
      y: interpolate(p, [0, 1], [680, 420]),
      o: (1 - i / 6) * 0.35,
    };
  });

  return (
    <svg
      width={WIDTH}
      height={HEIGHT}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      style={{ position: "absolute", inset: 0 }}
    >
      {/* goal line + posts, minimal */}
      <line x1="1320" y1="260" x2="1320" y2="460" stroke={theme.border} strokeWidth={6} />
      <line x1="1320" y1="260" x2="1560" y2="260" stroke={theme.border} strokeWidth={6} />
      <line x1="1320" y1="460" x2="1560" y2="460" stroke={theme.border} strokeWidth={3} opacity={0.4} />
      {/* penalty arc, minimal */}
      <path
        d="M 480 760 Q 640 560 480 420"
        stroke={theme.border}
        strokeWidth={3}
        fill="none"
        opacity={0.5}
      />
      {/* motion trail */}
      {trail.map((t, i) => (
        <circle key={i} cx={t.x} cy={t.y} r={16 - i} fill={theme.accent} opacity={t.o} />
      ))}
      {/* ball */}
      <circle cx={x} cy={y} r={22} fill={theme.text} />
      <circle cx={x} cy={y} r={22} fill="none" stroke={theme.border} strokeWidth={2} />
    </svg>
  );
};

const WIDTH = 1920;
const HEIGHT = 1080;

export const FreezeShot: React.FC = () => {
  const frame = useCurrentFrame();

  const ballProgress = interpolate(frame, [0, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const wordsVisible = Math.max(
    0,
    Math.min(HEADLINE_WORDS.length, Math.floor((frame - TYPE_START) / FRAMES_PER_WORD) + 1),
  );

  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <BallAndGoal progress={ballProgress} />
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            fontFamily: spaceGrotesk.fontFamily,
            fontSize: 92,
            fontWeight: 700,
            color: theme.text,
            letterSpacing: -1,
          }}
        >
          {HEADLINE_WORDS.slice(0, wordsVisible).join(" ")}
          {wordsVisible < HEADLINE_WORDS.length && wordsVisible > 0 ? (
            <span style={{ opacity: frame % 20 < 10 ? 1 : 0, color: theme.accent }}>|</span>
          ) : null}
        </div>
      </div>
    </div>
  );
};
