# Reference study — what "high production level" means here

Two reference videos in `video/references/` (gitignored, not ours to
redistribute), both turn out to be **HyperFrames' own promo films** —
`remix-polymarket-perps-result.mp4` (37.1s, 1280×720/24fps, cream/editorial)
and `remix-zaro-result.mp4` (68.5s, 1280×720/30fps, dark/software-demo).
That's useful: they're a direct preview of what the tool we're about to use
can actually produce, not an aspirational mood-board from a different stack.

Frames extracted with `ffmpeg` (contact sheets at ~1.5-2.3s intervals, plus
`select='gt(scene,0.12..0.35)'` scene-change detection) to
`video/references/frames/` (gitignored).

## Pacing

- **Polymarket** (cream): almost no hard cuts. Scene-change detection (even
  at a low 0.12 threshold) found only two real cuts before 27s, then a
  cluster of micro-changes at 27.0-28.2s (a flash-transition moment). The
  whole first ~18s reads as **one continuous camera move** through
  cross-dissolved/morphed content, not discrete shots — confirmed by the
  contact sheet showing 5+ distinct screens in that span with no detected
  cut between them.
- **Zaro** (dark): 26 scene changes across 68.5s → average shot ~2.6s, but
  not evenly spaced — mostly 1.5-3s per beat, accelerating into a rapid
  three-cut flash (65.0/65.07/65.13s) right before the outro.
- **Takeaway:** hard cuts are rare and are used as *punctuation* (a flash
  moment, a montage flourish), not the default scene-to-scene mechanism.
  Most transitions are caused by something on screen (an object growing
  into the next world, an iris opening, a morph) — confirmed by
  `.claude/skills/hyperframes/references/motion.md`'s own "Transitions
  with a cause" table, which names these same techniques.

## Typography

- Bold, confident geometric-humanist sans (rounded terminals — reads as
  something like General Sans/Switzer, not a heavy grotesque), large scale,
  high contrast against the background.
- **Word-level emphasis via a solid color badge**, not whole-line coloring:
  a keyword ("agents") sits inside a mint-green rounded-rect highlight,
  rest of the sentence plain. Confirmed at `pm_detail_tabs.jpg`.
- **De-emphasis of connector words**: "and" rendered dimmer/gray against
  bright keywords around it — creates reading rhythm (`zaro_detail_card.jpg`).
- **Inline UI chips at text baseline height**: a small rounded image
  thumbnail and a chat-bubble-shaped tag sit *inside* a sentence, between
  words, scaled to match the type size — not a separate screen
  (`zaro_detail_card.jpg`: "Files [image chip] and [Claude Code bubble]
  agents").
- One headline entrance uses a halftone **echo/ghost-trail** effect:
  2-3 duplicate offset copies in a green dot-pattern behind the sharp main
  text (`pm_detail_tabs.jpg`, "Start with a Hook").

## Color & texture

- Two "worlds," never mixed mid-scene: a soft cream background with
  diagonal pastel aurora/light-leak streaks, and a near-black background —
  both textured with a faint scattered dot/square "digital grain" (visible
  in both light and dark scenes, e.g. `zaro_detail_outro.jpg`'s corners).
- Never flat solid, never white-flash on a cut.

## Transitions & camera

- Floating app-UI mockup cards: soft large-radius shadow, rounded corners
  (~16-24px), sometimes stacked/layered — but per
  `.claude/skills/hyperframes/references/types/launch.md`, product UI is
  **never tilted in 3D** and never shown as **a small window in an empty
  canvas**; it's staged as its own "world" at a real, readable scale, then
  punched into (1.6-2x) for the relevant chart/field.
- An iris/circle-wipe (`clip-path: circle()`) opens from a clicked point
  onto the next scene (`zaro_detail_outro.jpg`).
- A branded teardrop-shaped cursor appears in the UI-mockup shots —
  HyperFrames' own synthetic-cursor convention (catalog: `cursor-zoom-follow`,
  `press-ripple`, `agent-cursor`).
- Clean wordmark outro: logomark + wordmark, tagline in a small pill badge
  beneath, "performed" (built/wiped in) rather than faded.

## What the HyperFrames skill itself warns against

Reading `.claude/skills/hyperframes/references/story.md` (`§8`) turned up
something important: several things visible in these exact reference
videos are **named as banned defaults** when nothing else is decided — "a
dark void with small things floating in it," "glow and particles," "the
cream-paper-serif-mono-label look," "everything fading and sliding up,"
"the docked logo." The lesson isn't to copy the references' surface look —
it's to borrow the *craft* (real-UI discipline, word-level typography,
transitions with a cause, a number landing with weight) while avoiding the
generic-AI-video tells the tool's own docs call out by name. Also flagged
there: **"typing into a prompt box as the opener, with nothing else on
screen, is the tell of a generated film."** Our approved hook does exactly
that — see the restructuring below.

The skill's canonical launch-film beat shape (`story.md §4`, scaled to our
~90s): **Hook ~8% → Tension ~12% → Reveal ~8% (product named, 20-30% in) →
Proof ~40% (real task, 3-6 verbs) → Payoff ~20% (the biggest visual moment,
once, 60-80% in) → Claim ~8% → End card ~8%, held 2-3s.**

## Proposed changes to pacing/staging (script text unchanged)

`SCRIPT.md`'s six lines stay exactly as approved — this only changes how
they're staged and where the wordmark/biggest-number moments land:

1. **Hook won't open on blank-canvas typing.** The real scoreboard
   establishes first (satisfies "open on the feature working on something
   recognisable" + avoids the named "typing in an empty canvas" tell); the
   headline builds around/over it, not alone.
2. **The "Sliding Doors" wordmark moves out of the hook** into its own
   brief Reveal beat (~18-25s) after Tension, built via the literal
   door-panels-parting motif (on-brand *and* a legitimate "object on
   screen causes the transition").
3. **Scenes 3+4 (product demo + ask-the-AI) merge into one continuous
   Proof sequence** of verbs — pick, change, watch it compute, ask (EN),
   ask (HE) — per the "3-6 verbs" shape, each isolated via a 1.6-2x push-in
   rather than a separate full scene cut.
4. **The 9%→92% divergence becomes its own Payoff beat**, pulled out to
   land once with real weight (a count-up stat, not buried mid-demo) around
   60-80% in, instead of sitting inside the middle of the product
   walkthrough.
5. **"Under the hood" is reframed as the Claim**, not a technical aside:
   the one-line thesis — *every number is real, two agents, nothing
   invented* — happens to be exactly what the agent diagram proves, so the
   beat does double duty instead of feeling like an interlude.
6. **Outro becomes a proper End card**: wordmark performed (built, not
   faded), repo/demo links land and hold.

Net runtime: still ~90s. Beat shares above are a target, not a hard
rule — I'll fit them to our real captured-footage lengths the same way
the original storyboard was audio-driven.

## Scene-by-scene plan: catalog blocks + borrowed technique

Catalog browsed via `npx hyperframes catalog --json` (448 entries: 210
blocks, 238 components) and filtered by relevance; exact names below are
confirmed to exist in the registry.

| Scene (beat) | Reference technique borrowed | Catalog blocks/components |
|---|---|---|
| **1. Hook** | Open on the real, working product (never blank canvas); headline builds around it, not alone | `app-showcase` or `browser-device-stage` to stage the real scoreboard capture at proper scale (flat, not tilted, not tiny); `headline-slam` or `kinetic-center-build` for the question headline; `marker-highlight` for emphasis. Large kinetic text *is* the caption here (word-synced to narration) — no caption bar. |
| **2. Tension** | "The old way failing on screen" — an unresolved disagreement, not a solved one | `text-state-swap` or `per-word-crossfade` to show two contradicting "opinions"; `strikethrough-replace` on one claim for the unresolved/dead-end feel |
| **2.5 Reveal** | Product named as a title, "performed rather than faded in," ~20-30% in | `panel-reveal` (the literal sliding-doors motif, repositioned from the old hook) or `svg-mask-reveal` revealing the wordmark; `iris-reveal`/`sdf-iris` as the bridging transition in |
| **3. Proof** | Real UI, staged as a world then punched into 1.6-2x per beat; cursor 4-7% of frame, enters from off-screen, click causes the next thing | `browser-device-stage` for the establishing shot; `push-in` + `cursor-zoom-follow` + `press-ripple` for each interaction (dropdown, Run, the Ask box); `typed-prompt` for the EN/HE questions typing in; `caption-pill-karaoke` (restyled to our palette) for the caption bar — **this is a screen-recording scene, so the caption bar is on** |
| **4. Payoff** | "A number lands... made physical," the biggest visual moment, once | `mk-progress-stat` (count-up "92%") or `chart-story`/`mk-line-graph` redrawing the real win-probability curve (real numbers from `data/featured_curves.csv`, match 1678 — not a mockup, just designed rather than a tiny captured screenshot) |
| **5. Claim** | The one-thing statement, proven by what's on screen | `flowchart` (Engine → Analyst → Narrator → user, matching the prior approval's "full flow, data visibly moving"); `state-chip-rail` for the tool-call chips; `agent-cursor` as a thematic touch |
| **6. End card** | Mark lands once, with weight, performed; CTA held 2-3s | `logo-outro` (as originally planned); `cta-lockup`/`cta-close` for the repo/demo links |

**Transitions used across the film** (3-5 kinds, per the "cause" rule —
matches the reference videos' own measured mix): the literal door-panel
reveal (2), an iris/circle wipe (2.5→3 bridge), push-in/zoom-through-button
within Proof, a whip or morph into Payoff, a performed wipe/fill into the
End card.

## Caption rule (per your instruction)

- **Scenes with large kinetic text on screen** (1 Hook, 2 Tension, 2.5
  Reveal, 5 Claim, 6 End card): that text *is* the caption, word-synced to
  the narration via our existing Azure word-boundary timestamps. No
  separate caption bar.
- **Scene 3 Proof only** (the real screen-recording footage): the caption
  bar appears, styled from `caption-pill-karaoke`, restyled to
  `DESIGN.md`'s palette/fonts.
- Scene 4 Payoff is graphics-only (the chart/stat), so it also gets
  kinetic-text captions, not a bar, unless real footage is cut back in.

## Carried over, unchanged

Per your instruction: `SCRIPT.md` (word-for-word), `DESIGN.md`'s palette
and fonts (Inter + Space Grotesk — HyperFrames only resolves its bundled
fonts by name, so these ship as `@font-face` files in `assets/fonts/`, see
`references/format.md`), the Azure voiceover + word-level timestamps
(`scripts/generate-voiceover-v2.mjs`), and the Playwright capture approach
(adapted to HyperFrames' `npx hyperframes capture` / real local-instance
recording rather than the old Remotion clip-trimming model).
