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
  league.py            # fictional clubs + their hidden "true" traits
  match_engine.py       # simulates a single match minute by minute
  season_generator.py   # schedules and simulates a full season
scripts/
  generate_season.py    # entry point: builds a league, generates a season, saves + summarizes it
data/
  *.csv                 # generated season data (matches, goals, red cards)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate a season

```bash
python scripts/generate_season.py
```

This creates a fictional 6-club league, simulates a full double
round-robin season, and writes the results to `data/`.
