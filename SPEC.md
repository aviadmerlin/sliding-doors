# SPEC — Sliding Doors: Counterfactual Match Intelligence

Hackathon: Microsoft x Premier League "Inside the Game"
Deadline: Oct 27, 2026

## Idea

A simulation engine that re-runs a football match from any moment, hundreds
of times, with one thing changed (a penalty scored, a red card not given).
It measures how much each moment changed the win probability, and an AI
agent (on Microsoft Foundry) answers fans' "what if" questions in natural
language.

## Data approach

- A fictional league of 6 clubs with fictional names. Each club has hidden
  "true" traits: attack strength, defense strength, behavior when trailing
  or leading, performance with 10 men, late-game fatigue.
- Generate a full synthetic SEASON of past matches, where these traits
  influence what happens.
- Estimate each club's traits from the season data with pandas (not by
  reading the hidden values).
- Validate: compare estimated traits vs. the hidden true traits.
- The win-probability simulation uses the ESTIMATED traits, with a
  goal-rate model that depends on game state (score, minute, red cards).
  Show uncertainty ranges, not just single numbers.

## MVP checklist (must work by Oct 27)

- [x] 1. League + season generator
- [x] 2. Trait estimation + validation
- [ ] 3. Win probability chart for one match, minute by minute
- [ ] 4. Top 3 "turning points", ranked by win-probability swing
- [ ] 5. One working "what if" question, end to end, via an agent on Foundry
- [ ] 6. Simple Streamlit UI, deployed on Azure

## Stretch goals (only if time allows)

- Fan vs analyst mode
- Hebrew + English

## Rules

- Synthetic data only — no real clubs, players, or logos.
- Public GitHub repo.
- Secrets only in `.env` (gitignored from the start).
