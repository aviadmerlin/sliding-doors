# Sliding Doors — Counterfactual Match Intelligence

Built for the Microsoft x Premier League "Inside the Game" hackathon.

"Sliding Doors" re-runs a football match from any moment, hundreds of
times, with one thing changed — a penalty scored, a red card not given —
and measures how much that moment shifted the win probability. An AI
agent answers fans' "what if" questions about the match in plain language.

See [SPEC.md](SPEC.md) for the full MVP checklist and rules.

## Project structure

```
src/
  league.py             # fictional clubs + their hidden "true" traits (and how traits drift between seasons)
  squad.py               # ~14-player squads per club (shot/card shares, hidden finishing skill)
  match_engine.py         # simulates a single match minute by minute, via shots and xG
  season_generator.py     # schedules and simulates multiple seasons
  trait_estimation.py     # recovers each club's hidden traits from the generated data (Poisson regression)
  win_probability.py      # vectorized Monte Carlo re-simulation -> win/draw/loss probability from any match state
  counterfactual.py       # re-simulates a match with one real event changed (shot/goal/red card)
scripts/
  generate_season.py      # entry point: builds a league + squads, generates seasons, saves + summarizes
  estimate_traits.py       # estimates current traits, sweeps recency half-lives, validates, plots
  win_probability.py       # picks the most dramatic match, plots its win-probability curve + turning points
  calibration.py            # checks predicted win probabilities against actual outcomes (in- and out-of-sample)
  counterfactual_demo.py    # demos the counterfactual engine on 3 "what if" scenarios
  precompute_featured_matches.py  # fits the model once, precomputes curves for ~11 matches so the app loads instantly
data/
  *.csv                   # generated season data (matches, shots, goals, red cards, players, true/estimated traits)
  featured_curves.csv, featured_turning_points.csv, featured_matches.json  # precomputed for instant loading
assets/
  trait_validation.png     # estimated vs. true traits scatter (see below)
  win_probability.png      # win probability chart for the most dramatic match (see below)
  calibration.png           # predicted vs. actual win rates, out-of-sample (see below)
  counterfactual_demo.png   # actual vs. counterfactual curves for 3 "what if" scenarios (see below)
app.py                    # interactive Streamlit UI (see below)
.streamlit/config.toml    # dark sports-product theme
.claude/skills/developing-with-streamlit/  # official Streamlit agent skill (vendored, see Setup)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repo vendors the official [Streamlit agent skill](https://github.com/streamlit/agent-skills)
under `.claude/skills/developing-with-streamlit/` — Claude Code picks it up
automatically for any Streamlit work in this project (styling, caching,
layout, etc.).

## Generate seasons

```bash
python scripts/generate_season.py
```

Creates a fictional 20-club league (each with hidden traits and a squad
of ~14 players), simulates 5 seasons of a double round-robin, and writes
matches/shots/goals/red_cards/players/true_traits to `data/`.

## Estimate traits

```bash
python scripts/estimate_traits.py
```

Estimates each club's current traits from the shot-level data using a
weighted Poisson regression, tries several recency half-lives, and
validates the estimates against the known true traits for the latest
season. Saves `data/estimated_traits.csv` and the chart below.

### Estimated vs. true traits (latest season, best half-life = no decay)

![Estimated vs true traits](assets/trait_validation.png)

Attack and defense are recovered almost exactly (r = 0.96–0.97) — there's
a shot on goal almost every few minutes, so there's a lot of signal.
`protect_lead_factor` (r = 0.86) and `fatigue_factor` (r = 0.64) are
decent. `comeback_factor` (r = 0.32) and `ten_men_resilience` (r = 0.59)
are the hardest.

`comeback_factor` and `protect_lead_factor` share one degree of freedom
(trailing and leading are literally the same event seen from each side),
but that only shifts their *absolute level* by one shared constant — it's
gauge-invariant and verified not to affect correlation at all. The real
reason `comeback_factor` is harder: its true values vary less across
clubs (std 0.08 vs. 0.11), *and* it's estimated more noisily, because a
club's trailing minutes come from a much more varied mix of situations
(bad luck, tough opponents, off days — correlation with the club's own
attack is only -0.37) than its leading minutes, which are strongly
self-reinforcing (correlation with own attack is +0.70: good attackers
simply lead a lot, giving a cleaner signal). `ten_men_resilience` is
estimated from red cards, which are rare (~12 per club across 5 seasons).

Shrinkage (pulling each club's estimate toward the league average,
weighted by how much data backs it) lowers the error for the two
noisiest traits — `comeback_factor` MAE 0.074 → 0.061, `ten_men_resilience`
MAE 0.095 → 0.059 — but makes the two already-solid traits slightly
worse, so it's applied only where it helps.

"No decay" (pooling all 5 seasons equally) won because traits only drift
slowly season to season — with a small, gradual random walk and just an
occasional bigger jump, more pooled data reduces noise more than recency
weighting helps. If clubs changed faster, a shorter half-life would win
instead.

## Win probability

```bash
python scripts/win_probability.py
```

Screens every match of the latest season for the biggest win-probability
swing (a fast, coarse Monte Carlo pass), then re-simulates the winner
minute by minute: at each minute, replay the rest of the match thousands
of times from the real game state (using the estimated traits), and tally
how often each outcome happens. Fully vectorized with numpy — thousands
of replays advance one simulated minute at a time as array operations,
not a Python loop per replay.

![Win probability](assets/win_probability.png)

The selected match (picked automatically, not hand-picked) was a 3-2
thriller: the eventual winner went behind in the 21st minute and sat at
under 5% win probability for most of the second half, before an
87th-minute equalizer (which mostly swung the *draw* probability, not
the win probability) and an 88th-minute winner completed the turnaround.
Turning points are ranked by total movement across all three outcome
probabilities (home/draw/away), not just the home-win number — otherwise
a goal that mainly makes a draw likely wouldn't register as pivotal even
though it clearly is.

**The uncertainty bands reflect two different things stacked together.**
Early versions only captured simulation noise — "if I replay this 3000
times, how much does the result wobble just from randomness?" That
understates the real uncertainty, because it treats the club traits as
if we knew them exactly, and we don't (see the correlations above). Now
each batch of replays uses a different set of traits, sampled from the
regression's own fitted uncertainty (`trait_estimation.sample_trait_draws`,
using the GLM's coefficient covariance — much cheaper than bootstrapping
by refitting, which would take minutes instead of seconds). Pooling
across batches means the spread captures "we're not sure Silverlake's
attack is exactly 1.3" on top of ordinary match randomness. At matched
trial budgets, bands come out roughly 50% wider during the uncertain
middle of a match than simulation noise alone would give — and converge
to the same tight band simulation-only bands give once the outcome is
essentially decided, which is the expected behavior.

## Calibration

```bash
python scripts/calibration.py
```

A win probability number is only useful if it means what it says: among
all the times the model said "70% chance of a home win," did the home
team actually win about 70% of the time? This checks exactly that, across
2,280 (match, minute) checkpoints from every match in the latest season
(stopping at minute 75, since predictions right before full time are
nearly trivial and wouldn't be a fair test).

**The honest version of this test fits traits on seasons 1-4 only and
checks calibration on season 5** — out-of-sample, the way it would
actually work in production (you'd never have next season's results
when making this season's predictions). Fitting and testing on season 5
alike, like the first version of this check did, lets the model get
credit for having already seen the answers.

![Calibration](assets/calibration.png)

Expected calibration error (the average gap between predicted and
actual, weighted by bucket size) is **0.026 out-of-sample**, versus
**0.020 in-sample** (`assets/calibration_in_sample.png`) — a small but
real gap from the leakage, in line with how slowly traits drift season
to season (see "no decay" winning the half-life sweep above). Both are
still good: buckets sit close to the diagonal within their error bars,
a reassuring sign that the estimated traits, even with the noise
discussed above, combine into well-calibrated match-level predictions.

## Counterfactual engine

```bash
python scripts/counterfactual_demo.py
```

A clean, reusable building block (no AI yet — this is what an agent will
call as a tool in a later step): given a real match and ONE change —
a missed shot/penalty becomes a goal, a goal is cancelled, or a red card
didn't happen — it re-simulates forward from that exact moment and
returns both the actual and counterfactual win-probability curves plus
final outcome probabilities. See `Counterfactual` / `CounterfactualResult`
in `src/counterfactual.py` for the exact (documented) input/output shapes.

Internally, the counterfactual branch is simulated as one coherent
imagined continuation (a single forward pass per sampled trait draw,
snapshotted at every minute) rather than independently re-guessing at
each checkpoint — the real match's continuation is known, so
`win_probability_curve` can afford to restart from the real state each
time, but a hypothetical future isn't known and has to be one consistent
storyline.

![Counterfactual demo](assets/counterfactual_demo.png)

Three examples, two on the win-probability match above and one on a
different match that actually had a red card (the first one didn't):

- **Cancel the 87th-minute equalizer** — the match was actually won 100%
  (it happened). Without it, the chance of anything but an away win drops
  to 5%: there wouldn't have been time left to recover.
- **Make the 38th-minute big chance a goal instead** — much less decisive:
  with 52 minutes still to play, the outcome becomes close to a
  three-way toss-up (29% / 28% / 44%), unlike the late equalizer.
- **Undo a 17th-minute red card** (different match, actually lost 1-5) —
  flips the predicted outcome entirely, from 100% away win to a 51%
  home-win favorite. That sending-off looks like the single biggest
  factor in the result.

## Interactive UI

```bash
python scripts/precompute_featured_matches.py   # run once, before deploying/sharing the app
streamlit run app.py
```

Ties every piece above into one app: pick a match (defaults to the most
dramatic one), see its score and event timeline, an interactive win
probability chart with uncertainty bands and turning points marked, and
a "What if...?" section that runs the counterfactual engine live — pick
an event from the match's timeline and a change, and it plots the actual
vs. counterfactual win probability side by side with the outcome deltas.
A clearly-marked "coming soon" placeholder sits where a free-text
natural-language question box will go once the Foundry agent is wired up.

**Cold start.** Fitting the trait regression from 5 seasons of
shot-level data takes ~25-30s, and simulating one match's win-probability
curve takes another ~5-10s — far too slow for a judge's first click to
sit on a blank page. `scripts/precompute_featured_matches.py` fits the
model once offline, ranks every season-5 match by win-probability swing,
and saves full curves + turning points for the most dramatic match plus
10 other high-swing games to `data/featured_curves.csv` /
`featured_turning_points.csv` / `featured_matches.json`. The app loads
those with a plain CSV/JSON read — no simulation, no model fit — so the
default view (and any of the 11 ⚡-marked matches in the picker) appears
in about a second.

Any *other* match, or a counterfactual on *any* match, still works — it
just triggers the trait-model fit on first use (`@st.cache_resource`,
so it only ever happens once per server process) and that match's
simulation (`@st.cache_data`, so revisiting it later is instant), both
with a clear spinner message rather than a silent wait.
