import { createTikTokStyleCaptions, type Caption } from "@remotion/captions";
import { useMemo } from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { inter } from "./fonts";
import { theme } from "./theme";

/**
 * Karaoke-style word-level captions: current word highlighted in accent
 * green and scaled up, spoken words in the normal text color, upcoming
 * words dimmed. Max ~2 lines via a constrained width. See DESIGN.md
 * "Captions". Pages/grouping from @remotion/captions'
 * createTikTokStyleCaptions; per-word color is custom (BasicCaptions has
 * no word-level highlight, which this spec requires).
 */
export const KaraokeCaptions: React.FC<{
  captions: Caption[];
  combineTokensWithinMilliseconds?: number;
}> = ({ captions, combineTokensWithinMilliseconds = 1200 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const currentMs = (frame / fps) * 1000;

  const { pages } = useMemo(
    () => createTikTokStyleCaptions({ captions, combineTokensWithinMilliseconds }),
    [captions, combineTokensWithinMilliseconds],
  );

  const page = pages.find(
    (p) => currentMs >= p.startMs && currentMs < p.startMs + p.durationMs,
  );
  if (!page) return null;

  // Fade the whole caption plate in/out at page boundaries.
  const sinceStart = currentMs - page.startMs;
  const untilEnd = page.startMs + page.durationMs - currentMs;
  const opacity = Math.min(
    interpolate(sinceStart, [0, 120], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
    interpolate(untilEnd, [0, 120], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
  );

  return (
    <div
      style={{
        position: "absolute",
        left: 160,
        right: 160,
        bottom: 96,
        display: "flex",
        justifyContent: "center",
        opacity,
      }}
    >
      <div
        style={{
          maxWidth: 1500,
          background: `${theme.bgPanel}D9`,
          border: `1px solid ${theme.border}`,
          borderRadius: 18,
          padding: "22px 40px",
          fontFamily: inter.fontFamily,
          fontSize: 42,
          fontWeight: 600,
          lineHeight: 1.3,
          boxShadow: "0 16px 48px rgba(0,0,0,0.4)",
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          // Word gap lives on the flex container, not in each token's text,
          // so inline-block's whitespace-trimming-at-box-edges can't eat it.
          columnGap: "0.32em",
          rowGap: "0.1em",
        }}
      >
        {page.tokens.map((token, i) => {
          const isCurrent = currentMs >= token.fromMs && currentMs < token.toMs;
          const isPast = currentMs >= token.toMs;
          const scale = isCurrent
            ? interpolate(currentMs, [token.fromMs, token.fromMs + 60], [1, 1.08], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              })
            : 1;
          return (
            <span
              key={i}
              style={{
                display: "inline-block",
                color: isCurrent ? theme.accent : isPast ? theme.text : theme.textDim,
                transform: `scale(${scale})`,
              }}
            >
              {token.text.trim()}
            </span>
          );
        })}
      </div>
    </div>
  );
};
