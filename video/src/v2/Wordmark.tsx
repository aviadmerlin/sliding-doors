import { theme } from "./theme";
import { spaceGrotesk } from "./fonts";

export const Wordmark: React.FC<{ scale?: number }> = ({ scale = 1 }) => (
  <div
    style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      transform: `scale(${scale})`,
    }}
  >
    <div
      style={{
        fontFamily: spaceGrotesk.fontFamily,
        fontSize: 108,
        fontWeight: 700,
        color: theme.text,
        letterSpacing: -1.5,
        display: "flex",
        alignItems: "center",
        gap: 20,
      }}
    >
      <span>Sliding</span>
      <span style={{ color: theme.accent }}>Doors</span>
    </div>
  </div>
);
