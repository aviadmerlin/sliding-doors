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
  model_params.csv, model_cov.csv, model_team_draws.csv  # the fitted trait model, saved so the app never refits it
  model_league_constants.csv  # shot-rate/composition constants estimated from data (see "removing data leakage" below)
assets/
  trait_validation.png     # estimated vs. true traits scatter (see below)
  win_probability.png      # win probability chart for the most dramatic match (see below)
  calibration.png           # predicted vs. actual win rates, out-of-sample (see below)
  counterfactual_demo.png   # actual vs. counterfactual curves for 3 "what if" scenarios (see below)
app.py                    # interactive Streamlit UI (see below)
tests/                    # pytest suite: data integrity, model logic, consistency, home advantage/leakage
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

### Finishing skill and removing data leakage

A code review found two problems with how the win-probability engine
(`src/win_probability.py`) got its numbers: (1) it never modeled a club's
finishing quality at all, and (2) it imported several "true" constants
straight from `src/match_engine.py` — the match generator — instead of
estimating them, including home advantage even though the regression
already fits a `home` coefficient that was just sitting unused.

**Finishing skill** is now estimated per club as (recency-weighted) goals
divided by (recency-weighted) xG across its shots, shrunk toward the
no-bias baseline of 1.0 the same way the noisier traits are. It feeds
into the engine as `finishing_factor`, a multiplier on shot quality
alongside the usual attack/defense numbers.

**Every other constant the engine needs is now estimated from data, not
imported from the generator:** home advantage from the regression's own
`home` coefficient (and, unlike the other league-wide constants, it now
varies per sampled trait draw, so its uncertainty is reflected in the
app's confidence bands too); base shot rate from the regression's
intercept (corrected for the fact that the regression is fit on total xG,
not shot counts); and shot composition (penalty/big-chance shares, xG
means per shot type) and the red-card rate as plain empirical rates from
`shots.csv`/`red_cards.csv`. A test (`tests/test_home_advantage_and_leakage.py`)
parses `src/win_probability.py`'s own source to confirm it never imports
from `src/match_engine.py` at all.

| constant | estimated | true | relative error |
|---|---:|---:|---:|
| home advantage | 1.0960 | 1.1000 | 0.36% |
| base shot rate (per 90) | 10.90 | 11.00 | 0.92% |
| penalty share of shots | 0.01093 | 0.01100 | 0.65% |
| big-chance share of shots | 0.1211 | 0.1200 | 0.90% |
| penalty xG mean | 0.7599 | 0.7600 | 0.01% |
| big-chance xG mean | 0.2501 | 0.2500 | 0.05% |
| normal-shot xG mean | 0.0802 | 0.0800 | 0.24% |
| red card chance (per team-minute) | 0.000681 | 0.000700 | 2.68% |

Every estimate lands within a few percent of the true value it's standing
in for — close enough to be useful, far enough apart that it's clearly an
honest estimate rather than a disguised peek at ground truth.

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

The selected match (picked automatically, not hand-picked) was a 2-3
collapse-and-comeback: Stonebridge Town went 2-0 up inside the first
half-hour and sat above 80% win probability for most of the match —
before Harrow Vale clawed back three unanswered goals late on (57', 79',
89'), the last one completing the turnaround and flipping the away win
probability from under 2% to 98% in a single minute. Turning points are
ranked by total movement across all three outcome probabilities
(home/draw/away), not just the home-win number — otherwise a goal that
mainly makes a draw likely wouldn't register as pivotal even though it
clearly is.

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
actual, weighted by bucket size) is **0.0315 out-of-sample**, essentially
indistinguishable from **0.0289 in-sample**
(`assets/calibration_in_sample.png`) — any leakage effect is smaller
than the run-to-run noise at this sample size, consistent with how
slowly traits drift season to season (see "no decay" winning the
half-life sweep above). Both are good: buckets sit close to the diagonal
within their error bars, a reassuring sign that the estimated traits,
even with the noise discussed above, combine into well-calibrated
match-level predictions.

## Counterfactual engine

```bash
python scripts/counterfactual_demo.py
```

A clean, reusable building block (no AI yet — this is what an agent will
call as a tool in a later step): given a real match and ONE change —
a missed shot/penalty becomes a goal, a goal is cancelled, or a red card
didn't happen — it re-simulates forward from that exact moment and
returns the real match's win-probability curve, a counterfactual
forecast, and a fair before/after comparison. See `Counterfactual` /
`CounterfactualResult` in `src/counterfactual.py` for the exact
(documented) input/output shapes.

**Two things worth getting right, both of which an earlier version got
wrong:**

- **There's no single "counterfactual timeline."** Once a moment is
  changed, the rest of the match isn't one knowable story — it's a
  distribution over what might happen. An earlier version traced *one*
  simulated path per trait draw and read off "share of replays currently
  level or ahead" at each minute — which isn't win probability at all
  (right after a goal that ties the score, it reads as ~0%, then climbs
  only as that one path's noise happens to separate). The honest fix:
  simulate forward from the fork once (many replays, pooled across trait
  draws) to get the *one* well-defined number — probability of each
  outcome by full time — and hold it flat from the fork onward, with its
  uncertainty band. No fabricated trajectory.
- **Comparing against the real final result is hindsight, not a fair
  test.** The real result is certain (100%/0%/0%) because the match
  already happened; the counterfactual is an honest forecast. Comparing
  them makes every counterfactual look more dramatic than it is. The fix:
  compare *forecast to forecast*, both at the exact same minute — the
  real match's forecast right after what actually happened (e.g. right
  after the miss) against the counterfactual's forecast right after the
  hypothetical change (e.g. right after the goal). The real final score
  is still shown, but clearly labeled as what actually happened, not a
  number to compare against a forecast.

![Counterfactual demo](assets/counterfactual_demo.png)

Three examples, two on the win-probability match above and one on a
different match that actually had a red card (the first one didn't) —
each a same-minute, forecast-vs-forecast comparison:

- **Cancel the 89th-minute winner** — Harrow Vale's win forecast right
  after the real goal was 98%; without it (still 2-2 with a minute left),
  it collapses to 1%.
- **Make the 88th-minute near-miss a goal instead** — right before Harrow
  Vale's winner, Stonebridge Town's forecast (still 2-2) was just 4%; if
  their shot a minute earlier had gone in instead, making it 3-2, that
  jumps to 98% — the mirror image of the goal that actually decided it.
- **Undo a 36th-minute red card** (different match, won 4-0) — West
  Harbor City's forecast right after the real sending-off was 80%;
  without it, 67%. The card helped, but a 4-0 scoreline was clearly not
  down to the card alone.

## Interactive UI

```bash
python scripts/precompute_featured_matches.py   # run once, before deploying/sharing the app
streamlit run app.py
```

Ties every piece above into one app: a one-line explanation and 3-step
guide for first-time visitors, a match picker (defaults to the most
dramatic match), a score header, a small "club profile" card per team
(plain-language traits like "Strong at protecting a lead," with a note
on which estimates are solid vs. noisier), the event timeline, an
interactive win probability chart with uncertainty bands and turning
points marked, turning-point cards with a clear before → after
(`Bridgeport win: 34% → 78%`) plus a one-line plain-English summary, and
a "What if...?" section that runs the counterfactual engine live —
pick an event from the match's timeline (the change dropdown only ever
shows the one change that event actually supports) and it shows a fair
before → after at the fork minute (e.g. "Harrow Vale win: 98% → 1%"),
the real match's curve against the counterfactual's honest flat
forecast + band, and the real final result in its own clearly-labeled
panel. A clearly-marked "coming soon" placeholder sits where a free-text
natural-language question box will go once the Foundry agent is wired up.

**Cold start and the fitted model.** Fitting the trait regression from 5
seasons of shot-level data takes ~15-20s — far too slow for a judge's
first click, and wasteful to redo on every server start besides.
`scripts/precompute_featured_matches.py` fits it once offline and saves
the result to `data/model_params.csv` / `model_cov.csv` /
`model_team_draws.csv`; the app always *loads* the model from disk
(a fast CSV read) instead of refitting it. The same script ranks every
season-5 match by win-probability swing and saves full curves + turning
points for the most dramatic match plus 10 other high-swing games to
`data/featured_curves.csv` / `featured_turning_points.csv` /
`featured_matches.json`. Net effect: the default view (and any of the 11
⚡-marked matches) loads in about a second, any other match simulates in
a couple of seconds the first time (`@st.cache_data`, instant after
that), and even the very first counterfactual of a session — on any
match — now takes well under a second of actual computation, down from
~15s when the app used to refit the model on demand.

**Turning points, fixed two ways.** Early versions occasionally flagged
a minute with no real event as a "turning point" — that's just Monte
Carlo noise between checkpoints, not a real turning point, so
`rank_turning_points` now only ever considers minutes with an actual
goal or red card. When two events land in the same minute (a goal and a
missed chance, say), the state-changing one is always attributed as the
cause — a missed shot never changes the simulated score, so it can't be
what drove a swing, even if it coincidentally happened in the same
minute; the other event is still shown as secondary context.

**Smoother curves via common random numbers.** The win-probability curve
used to independently re-simulate from scratch at every single-minute
checkpoint, so adjacent minutes could show unrelated Monte Carlo noise
even when nothing in the match had actually changed. Now every
checkpoint's simulation is reseeded from the *same* seed, so neighboring
checkpoints share most of their random draws — the curve only moves
where the real state actually changed, not from sampling noise. Bumping
replay counts alongside this (120 → 200 trials per trait draw) smooths
it further.
