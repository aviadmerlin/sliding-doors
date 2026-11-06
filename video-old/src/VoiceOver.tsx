import { Audio } from "@remotion/media";
import { staticFile } from "remotion";

export const VoiceOver: React.FC<{ file: string }> = ({ file }) => (
  <Audio src={staticFile(`audio/${file}`)} />
);
