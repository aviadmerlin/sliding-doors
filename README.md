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
scripts/
  generate_season.py      # entry point: builds a league + squads, generates seasons, saves + summarizes
  estimate_traits.py       # estimates current traits, sweeps recency half-lives, validates, plots
  win_probability.py       # picks the most dramatic match, plots its win-probability curve + turning points
data/
  *.csv                   # generated season data (matches, shots, goals, red cards, players, true/estimated traits)
assets/
  trait_validation.png     # estimated vs. true traits scatter (see below)
  win_probability.png      # win probability chart for the most dramatic match (see below)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
