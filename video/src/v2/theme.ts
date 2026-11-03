// Palette pulled 1:1 from the app's own .streamlit/config.toml -- see
// video/DESIGN.md "Palette" for the full rationale/table.
export const theme = {
  bg: "#0B1220",
  bgPanel: "#141C2E",
  bgSidebar: "#0F1729",
  border: "#243049",
  text: "#E5E7EB",
  textDim: "#9CA3AF",
  accent: "#22C55E", // green
  loss: "#EF4444", // red
  info: "#3B82F6", // blue
  amber: "#F59E0B",
  purple: "#8B5CF6",
} as const;

export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;
