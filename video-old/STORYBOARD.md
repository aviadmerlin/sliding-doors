# Storyboard — corrected against the real app

Target runtime: **~92.5s** (final hold trimmed to 2.5s per approval —
see "Running total" below). Narration drives scene length — each
scene's duration is its spoken line(s) at ~150 words/min plus a
deliberate silent visual beat where noted. Total script: **145 words
≈ 58s of speech**, leaving ~34-35s of pure visual/transition time
spread across the 6 scenes (chart divergence holds, diagram settle,
door-wipe transitions, final hold) — this is intentional pacing, not
dead air; flag if you want it more narration-dense.

Corrections from the draft, with reasons:

- **"Under the hood" is now the agent architecture, not the
  calibration chart.** The old video used a static calibration.png
  insert here; the new storyboard drops it in favor of animating the
  real Analyst/Narrator pipeline — a stronger fit for a hackathon
  judged partly on "hero technology" (Foundry, Agent Framework,
  multi-agent design) and reuses real, already-wired data instead of a
  static chart.
- **Exactly two agents**, not an open-ended "agents" diagram —
  `src/agent.py` has exactly Analyst + Narrator. The diagram shows
  precisely that, no more.
- **Dropped the "honest refusal" demo beat** from the old video (the
  third agent question, "what if they'd subbed at halftime?") to keep
  "Ask the AI" tight at 15s for one EN + one HE question. Easy to add
  back as a stretch beat if you want it — flag it in your feedback and
  I'll slot it into scene 4 or extend the runtime slightly.
- **Same real match and numbers as before** (Oakhaven City 3–2 Dunmore
  Town, match 1678, season 5 — still the app's default/featured match,
  data hasn't changed): losing 2-1 for 40+ minutes, chances under 10%,
  equalizer at 84', winner at 90'; cancelling the 84' goal gives the
  real re-simulated **9% → 92%** swing (validated twice now against
  live runs). The EN/HE agent questions are the same two validated in
  the first video (Dunmore's 26th-minute big chance in English; the
  90th-minute goal disallowed in Hebrew) — both produced clean,
  grounded answers against live data.

## Scene 1 — Hook (0:00–0:07, 7s)

**Visual:** Cold open, no white ever. A single stylized match moment:
a minimal pitch/ball graphic freezes mid-shot on the animated dark
background (not live UI — pure motion graphic). The line "What if this
shot went in?" types in word by word, Space Grotesk, center frame.
Beat. The frame "rewinds" (a quick reverse-scrub visual flourish — time
markers ticking backward). Then it splits down the middle into two
sliding panels — literal "sliding doors" — revealing the product
wordmark behind them as they part.

**VO (starts ~1.5s in, under the typing text):**
> "Every fan's had this argument — if that shot had gone in, we'd have won."

(15 words, ~6.0s spoken; remaining ~1.0s is the silent door-split reveal into scene 2)

## Scene 2 — Problem (0:07–0:15, 8s)

**Visual:** Through the opened doors: a quick, light montage — two or
three small stylized "chat bubble" opinions overlapping/arguing (no
real UI yet), then they dissolve as the Sliding Doors wordmark/logo
settles and the floating browser window scales in from behind it,
still showing a dark loading-free state (not the live app yet — the
browser chrome is visible, empty/branded).

**VO:**
> "It's always just opinion. Nobody's ever been able to prove it — until now."

(14 words, ~5.6s; remaining ~2.4s is the logo-to-browser-window settle, into scene 3's live capture)

## Scene 3 — Product in action (0:15–0:50, 35s) — THE WOW MOMENT

**Visual (real Playwright capture, local instance, inside the floating
browser window, virtual camera throughout — never static full-screen):**
1. Camera on the scoreboard (Oakhaven City 3–2 Dunmore Town) — brief push-in.
2. Pan down to the win-probability chart + a quick staggered glance across
   the "Top 3 turning points" cards (42', 84', 90').
3. Pan/cut to the "What if...?" panel. Synthetic cursor opens the Event
   dropdown, clicks the 84' goal, clicks "Run counterfactual" — click
   ripple on each.
4. **Speed-ramp** the real ~7s computation wait (shown sped up, not cut
   entirely — it's the proof the engine is really computing, per the
   existing video's reasoning) into ~2s.
5. Camera pushes into the result panel and the divergence chart as the
   two curves (real vs. counterfactual) visibly pull apart — this is
   the beat the camera move exists for.

**VO:**
> "Oakhaven City were losing for over forty minutes — their chances under ten percent. Then two late goals flipped everything.
>
> Sliding Doors doesn't just show you that moment — it replays it. Cancel that eighty-fourth-minute equalizer, and the engine re-simulates the rest of the match, hundreds of times.
>
> Dunmore's chances jump from nine percent... to ninety-two."

(56 words, ~22.4s spoken; remaining ~12.6s covers the camera moves, the sped-up compute beat, and a held silent moment on the divergence chart with the positive chime SFX)

## Scene 4 — Ask the AI (0:50–1:05, 15s)

**Visual:** Door-wipe transition into the "Ask a what-if question" box.
Camera on the input as text types in (synthetic typing, matching real
capture timing) — English question first, answer appears (camera
holds on the answer text, captions carry it). Quick language-dropdown
switch (cursor click), Hebrew question types in (RTL), Hebrew answer
appears.

**VO:**
> "Fans can just ask. An agent built on Microsoft Foundry turns the question into a real simulation — in English, or Hebrew."

(22 words, ~8.8s; remaining ~6.2s is reading time held on each real answer — EN then HE — which is the visual payload of this beat)

**Questions used** (both previously validated live):
- EN: *"What if Dunmore had scored their big chance in the 26th minute?"*
- HE: *"מה היה קורה אם השער של אוקהייבן בדקה ה-90 היה נפסל?"*

## Scene 5 — Under the hood (1:05–1:20, 15s)

**Visual:** Door-wipe into a clean dark canvas (motion graphic, no
captured UI). Four stages stagger in left to right: **Simulation
Engine** box, **Analyst** (blue), **Narrator** (purple), **fan icon**
(green) — Analyst/Narrator labeled "Microsoft Foundry · gpt-5-mini".
Edges draw on, then data visibly travels them on a continuous loop: an
amber dot Analyst ↔ one of four tool pills (carrying a tiny result
label like `9% → 92%` on the way back), then a green dot
Analyst → Narrator → the fan icon. See DESIGN.md "Agent diagram" for
the full spec — this is the full real pipeline, not simplified.

**VO:**
> "Under the hood, two agents work together on Microsoft Foundry — an Analyst that calls the simulation, and a Narrator that explains it. Every number's grounded — nothing's invented."

(29 words, ~11.6s; remaining ~3.4s lets the diagram finish settling in silence before the next wipe)

## Scene 6 — Outro (1:20–1:30+, 10s + 2.5s hold)

**Visual:** Door-wipe into the final title card: wordmark center
frame, tagline beneath, then the repo link and live-demo link fade in
staggered below. Everything finishes entering by ~1:27, then **holds
static for 2.5s** with no motion before the video ends (trimmed from
the original 3-4s brief to land the total runtime near 90s).

**VO:**
> "Sliding Doors. Every match, every moment, every what if."

(9 words, ~3.6s; the rest of the scene is the silent hold)

---

## Running total

| Scene | Window | Spoken | Visual-only |
|---|---:|---:|---:|
| 1 Hook | 7s | 6.0s | 1.0s |
| 2 Problem | 8s | 5.6s | 2.4s |
| 3 Product | 35s | 22.4s | 12.6s |
| 4 Ask AI | 15s | 8.8s | 6.2s |
| 5 Under the hood | 15s | 11.6s | 3.4s |
| 6 Outro | 10s + 2.5s hold | 3.6s | 8.9s |
| **Total** | **~92.5s** | **58s** | **34.5s** |

**Approved 2026-11-02:** final hold trimmed from 3-4s to 2.5s, landing
at ~92.5s — close enough to "around 90s" per approval; no further
trims applied.
