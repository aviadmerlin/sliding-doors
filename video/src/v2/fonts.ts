// Two Google Fonts, matching the app's own Inter plus one display face --
// see video/DESIGN.md "Typography".
import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadSpaceGrotesk } from "@remotion/google-fonts/SpaceGrotesk";

export const inter = loadInter("normal", {
  weights: ["400", "500", "600", "700"],
  subsets: ["latin"],
});

export const spaceGrotesk = loadSpaceGrotesk("normal", {
  weights: ["500", "700"],
  subsets: ["latin"],
});
