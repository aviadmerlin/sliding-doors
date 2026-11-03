# Narration script (v2 — re-cut to the 7-beat structure)

Re-cut from the original 6-line, 145-word script to match the restructure
approved against `REFERENCES.md` (Hook → Tension → Reveal → Proof →
Payoff → Claim → End card). Same facts, same real numbers (still match
1678, Oakhaven City 3–2 Dunmore Town, the real 9%→92% swing) — shorter,
punchier lines, and the big number now gets its own dedicated Payoff line
instead of being buried mid-demo.

**Why so much shorter**: neither reference video has an audio track at
all (checked via `ffprobe` — zero audio streams, not even music). Their
"high production" feel comes entirely from kinetic type and motion
craft, not narration pacing. 87 words total now (was 145) — more of the
story is carried by what's on screen, matching `story.md`'s "show, don't
list."

Voice unchanged: Azure AI Speech, `en-US-AndrewNeural`. Timing estimates
below use our actually-measured delivery rate (~175 wpm / 2.92 words/sec,
from the real hook-line audio in the v1 build), not the generic 150wpm
planning assumption — still re-measured from real audio once each line is
actually synthesized, same as before. 87 words total, ~29.8s estimated
spoken (verified by script, see "Running total").

---

## 1. Hook — target window ~7s

> What if that shot had gone in?

7 words ≈ 2.4s spoken. Leads with the question — the "fans arguing" idea
now lives entirely in Tension, not duplicated here. Opens on the **real,
fully-loaded scoreboard** (no loading state, not one frame — see capture
note at the end), headline builds around it.

## 2. Tension — target window ~8-9s

> It's always just opinion. Two fans, same goal, two different stories.

11 words ≈ 3.8s spoken. Visual: two contradicting claims shown clashing,
unresolved — the "old way failing on screen."

## 3. Reveal — target window ~6-7s

> Until now. This is Sliding Doors.

6 words ≈ 2.1s spoken. The shortest, hardest-landing line in the script —
door panels part, wordmark performed in. Visual carries this beat, not
narration.

## 4. Proof — target window ~34-36s

Four short verb-anchored lines, one per real action, spoken as each
happens — not one long paragraph:

> Oakhaven were losing for forty minutes. Pick the moment that changed it.

> Cancel the goal — the engine re-simulates the match, hundreds of times.

> Or just ask — in English,

> or Hebrew.

32 words ≈ 11.0s spoken, spread across the beat. The rest of the window
is real captured interaction (the dropdown, the Run click, the sped-up
compute, the EN/HE questions typing and their real answers appearing) —
the real UI and real agent replies carry meaning here, not continuous
narration.

## 5. Payoff — target window ~5-6s

> Dunmore's chances of winning: nine percent... to ninety-two.

8 words ≈ 2.7s spoken. The one dedicated line for the biggest number in
the film — held back from Proof specifically so it lands once, alone,
with full weight (count-up stat treatment, not a captured screenshot).
**Proof must visually establish what changed** (the 84th-minute goal
being cancelled, specifically) clearly enough that this jump reads as
caused, not just dramatic — see Proof's beat 2 below.

## 6. Claim — target window ~7s

> Two agents, on Microsoft Foundry. Every number comes from the model, never made up.

14 words ≈ 4.8s spoken. Changed from "every number's real" — the data is
synthetic (the README says so explicitly), so "real" contradicts that.
"Comes from the model, never made up" is the accurate claim: the *numbers*
are never invented by the agent, not that the underlying match data is
real-world. Keeps the Microsoft Foundry name-check (judged criterion).

## 7. End card — target window ~7s + 2.5s hold

> Sliding Doors. Every match, every moment, every what if.

9 words ≈ 3.1s spoken. Unchanged from v1 — already short and already
works as a sign-off line.

---

## Running total

| Beat | Target window | Spoken |
|---|---:|---:|
| 1 Hook | 7s | 2.4s |
| 2 Tension | 8.5s | 3.8s |
| 3 Reveal | 6.5s | 2.1s |
| 4 Proof | 35s | 11.0s |
| 5 Payoff | 5.5s | 2.7s |
| 6 Claim | 7s | 4.8s |
| 7 End card | 7s + 2.5s hold | 3.1s |
| **Total** | **~79s + hold ≈ 91.5s** | **29.8s** |

Windows are targets, not a hard rule — same as v1, final scene lengths
get driven by real measured audio + real captured footage length once
built, and will likely shift a few seconds either way.

## Capture note (your requirement #2)

The Hook's scoreboard shot must be a clean, fully-settled capture — no
spinner, no skeleton state, not for one frame. Build plan: navigate with
`waitUntil: networkidle` (already our pattern), then an explicit
post-load settle wait before the capture window starts, then a visual
check of the first captured frame specifically for any loading indicator
before it's used — not just trusting the network-idle signal, since
Streamlit can finish network activity before its UI has fully painted.
