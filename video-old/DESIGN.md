# Sliding Doors demo video — visual design system

One consistent system, derived directly from the app's own theme
(`.streamlit/config.toml`) so the captured UI never feels like a
different product from the motion graphics around it.

## Palette

Pulled 1:1 from the app's real theme — not reinvented:

| Token | Hex | Use |
|---|---|---|
| `bg` | `#0B1220` | Canvas background (deep navy, near-black) |
| `bgPanel` | `#141C2E` | Cards, browser-chrome bezel, caption plate |
| `bgSidebar` | `#0F1729` | Secondary panels, diagram nodes |
| `border` | `#243049` | Hairlines, browser-chrome border, node outlines |
| `text` | `#E5E7EB` | Primary text, captions |
| `textDim` | `#9CA3AF` | Secondary labels (= app's `grayColor`) |
| `accent` (green) | `#22C55E` | Brand accent — primary CTA color, karaoke caption highlight, Oakhaven/positive swings, the literal "doors" motif |
| `loss` (red) | `#EF4444` | Negative swings, Dunmore when trailing |
| `info` (blue) | `#3B82F6` | Agent diagram — Analyst node |
| `amber` | `#F59E0B` | Agent diagram — tool-call data pulses |
| `purple` | `#8B5CF6` | Agent diagram — Narrator node |

Background is never flat: a slow, subtle radial/linear gradient
(`bg` → `bgSidebar`) with a faint animated grid or particle drift
(opacity ≤ 0.06) keeps every non-captured frame alive without
competing with the UI. **Never white** — every transition moves
between dark states; no crossfade is allowed to pass through white.

## Typography

Two Google Fonts, matching the app's own `Inter` plus one display face:

- **Inter** (400/500/600/700) — the app's own font. Used for captions,
  UI-adjacent labels, and body copy, so captured screen and graphics
  read as the same product.
- **Space Grotesk** (500/700) — display headlines only: the hook line,
  section titles ("Under the hood"), the outro title card. Geometric,
  confident, distinct from Inter at a glance so big type moments read
  as *statements*, not UI.

Caption sizing follows the video-layout safe-area rule (scaled to
1920×1080): body captions ≥ 40px, section headlines ≥ 84px (scaled
~150px), minimum 80px/100px margins from the frame edge.

## Motion style

- **Easing:** confident and snappy, not bouncy. Entrances use
  `spring()` with moderate damping (settles in ~12-15 frames, one
  small overshoot at most); exits ease out linearly. This is a
  data-intelligence product, not a toy — motion should feel precise.
- **Staggered entrances:** multi-element moments (turning-point cards,
  agent nodes, outro links) enter with a ~4-6 frame stagger per item,
  never all at once.
- **Transitions between scenes:** a literal **sliding-door wipe** —
  two panels slide together/apart from the center, matching the
  product name — used at every scene cut, 12-15 frames, dark-to-dark.
  A short whoosh SFX (placeholder slot, see below) plays under each.
- **Hold:** the final outro frame holds 3-4s with no motion once its
  entrance finishes, before the video ends.

## The floating browser window

The captured app footage is never full-bleed/static. It sits inside a
minimal browser-chrome card:

- Rounded rect, `bgPanel` fill, 1px `border` stroke, 20px corner radius.
- Thin top address-bar strip (36px): three small `textDim` traffic-light
  dots (left) + a pill showing the URL being demoed (`sliding-doors`,
  not the raw localhost address — see Capture plan), centered.
- Drop shadow: soft, large-radius, tinted toward `accent` at low
  opacity (a subtle green glow, not a generic black shadow) — ties the
  chrome to the brand without shouting.
- The card is never pinned full-frame: it's scaled to roughly 80-90%
  of frame width/height with margin to the animated background
  showing on all sides, so the background is always visible as context.

## Virtual camera

Pan/zoom keyframes target specific regions of the captured footage
(scoreboard, the Event dropdown, the counterfactual result panel, the
chart's divergence point), driven by element bounding boxes logged
during the Playwright capture pass (`scripts/record-footage.mjs` will
be extended to dump `{selector, x, y, width, height, timestampMs}` per
beat, not just timestamps). Camera moves are slow, deliberate pushes
(scale 1.0 → 1.08-1.15 over 20-30 frames), never a snap-zoom.

## Synthetic cursor

A drawn cursor (not the OS cursor, which Playwright headless capture
doesn't render anyway) overlays the footage, driven by the same
recorded (x, y, timestamp) click log. On each click: a small filled
circle cursor + an expanding, fading ring ("click ripple") in `accent`
green, ~15 frames. Cursor moves between clicks with a short ease, not
a teleport.

## Captions

- `@remotion/captions`, karaoke/word-level highlight: current word in
  `accent` green + slightly scaled up (1.0 → 1.06), spoken/past words
  in `text`, upcoming words dimmer (`textDim`).
- Max 2 lines, centered, bottom-third, on a `bgPanel`-at-85%-opacity
  rounded plate for contrast against any background (captured UI or
  graphics).
- On screen for the **entire** video, including the hook and outro
  (even short graphic-only beats get a caption if there's a line of
  narration playing; pure-visual silent beats show no caption).
- Word timestamps come from the Azure Speech SDK's native
  `wordBoundary` event (confirmed supported, SDK 1.53.0 already
  installed) — no Whisper needed.

## SFX (placeholders, see MUSIC/SFX note at the end)

Four short placeholder slots, wired with volume envelopes now,
swapped for real files later:
- Scene-transition whoosh (the sliding-door wipe)
- Click tick (synced to the synthetic cursor's click moments)
- A soft positive "chime" under the 9%→92% divergence reveal
- A low "settle" tick as each agent node appears in "Under the hood"

## Agent diagram ("Under the hood")

Four stages, left to right, the real architecture end to end — not
inflated to look busier, made to feel complete instead: **Simulation
Engine** (gray/border-tone box, labeled `win_probability.py` /
`counterfactual.py`) → **Analyst** (blue) → **Narrator** (purple) →
a small **fan icon** (green). Analyst and Narrator are both labeled
"Microsoft Foundry · gpt-5-mini" on a shared base plate, matching
`src/agent.py` exactly (Analyst calls tools and never writes the
reply; Narrator writes the reply and has no tools of its own).

To avoid feeling sparse, the flow is shown as **data actually moving**,
not just static arrows:

- Four small tool pills (`list_match_events`, `get_win_probability`,
  `run_counterfactual`, `get_club_profile`) sit between the Simulation
  Engine and the Analyst. An amber dot travels Analyst → a tool pill →
  back to Analyst with a tiny payload label (e.g. `9% → 92%`) riding
  along the return trip — this is the real tool-call/tool-result
  round trip.
- A second dot, green, travels Analyst → Narrator carrying a small
  "grounded numbers" tag, then Narrator → fan icon carrying a tiny
  chat-bubble glyph — the final answer reaching the user.
- Both dot-trips loop continuously (not once-and-done) for as long as
  the scene holds, on staggered timing so the diagram always has at
  least one pulse of motion in flight — this is what keeps a
  4-stage/6-node diagram from reading as sparse in a 15s scene.
- Edges draw on first (stroke-dashoffset, left to right) before any
  dot starts traveling, and nodes stagger in before their edges draw.
