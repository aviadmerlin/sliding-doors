"""Simulates a match (or part of one), minute by minute, via shots and xG.

Each minute, each team has a small chance of taking a shot. That chance
(the "shot rate") depends on:
  - the team's attack vs. the opponent's defense
  - a home-advantage bump
  - trailing/leading: a trailing team shoots more, and the leading team's
    opponent defends the lead (concedes fewer shots)
  - late-game fatigue: a tiring team concedes more shots after minute 75
  - a red card: ten_men_resilience weakens both the shot rate a team
    generates and the rate it concedes

Once a shot happens, it's classified as a penalty, a "big chance", or a
normal shot — each with its own xG range — and a player from the
shooting team's squad is picked (weighted by their shot_share). The shot
is a goal if a random draw lands below xG adjusted by that player's
hidden finishing skill. Red cards are likewise assigned to a player,
weighted by card_share.

On top of the club-level traits, each team also gets two things no club
trait captures: a hidden "day form" (how well THIS team is playing in
THIS specific match, unknown at kickoff and never revealed) and short
in-match "pressure" spells (a few minutes of being on top, then fading).
Both affect the SAME two channels every shot does -- the rate of shots
and the quality (xG) of each shot -- via one combined multiplier psi
split evenly (as sqrt(psi)) across the two channels, so psi is exactly
the effect on total xG output. That exact split is what lets
win_probability.py's Bayesian belief update (which only ever observes
total xG, never the rate/quality split) track the true value without
bias -- see its module docstring.

The engine can start from any game state (minute, score, red cards), so
later we can re-simulate a match from a specific moment with one thing
changed.
"""

import numpy as np

from src.league import Club
from src.squad import Player, pick_player

BASE_SHOT_RATE_PER_90 = 11.0  # expected shots per team per 90 min, at attack = defense = 1.0
HOME_ADVANTAGE = 1.1
RED_CARD_CHANCE_PER_MINUTE = 0.0007  # ~6% chance of a red card per team per match

PENALTY_SHOT_FRACTION = 0.011  # share of all shots that are penalties (~0.25/match combined)
BIG_CHANCE_FRACTION = 0.12     # share of non-penalty shots that are "big chances"

PENALTY_XG_MEAN = 0.76
BIG_CHANCE_XG_RANGE = (0.15, 0.35)
NORMAL_XG_RANGE = (0.02, 0.14)

# Day form: a per-match, per-team hidden multiplier psi on TOTAL xG output
# (shot rate x chance quality combined), psi ~ Gamma(k, 1/k) so E[psi] = 1.
# k (the "concentration") controls how much day form varies match to match --
# smaller k means wilder days. Applied to the engine as sqrt(psi) on each of
# the two channels (see module docstring).
DAY_FORM_SHAPE = 8.0

# Pressure spells: a team can enter a short "on top" spell (checked each
# minute while not already in one) and leaves it the same way (checked each
# minute while in one) -- a simple two-state Markov chain giving
# geometrically-distributed spell lengths, mean 1/PRESSURE_EXIT_CHANCE_PER_MINUTE
# minutes, active ENTRY/(ENTRY+EXIT) of the match on average. While pressing,
# a team's own psi is multiplied by PRESSURE_BOOST (again split sqrt/sqrt
# across the rate and quality channels) -- this is the one hidden-state
# effect the win-probability model deliberately does NOT try to track
# directly (see win_probability.py); it just shows up as extra noise around
# day form, and as real signal in the app's momentum chart.
PRESSURE_ENTRY_CHANCE_PER_MINUTE = 0.025
PRESSURE_EXIT_CHANCE_PER_MINUTE = 0.15
PRESSURE_BOOST = 1.25


def draw_day_form(rng: np.random.Generator) -> float:
    return float(rng.gamma(DAY_FORM_SHAPE, 1.0 / DAY_FORM_SHAPE))


def simulate_match(
    home: Club, away: Club, home_squad: list[Player], away_squad: list[Player],
    rng: np.random.Generator, start_minute: int = 0, home_goals: int = 0, away_goals: int = 0,
    home_red_card_minute: int | None = None, away_red_card_minute: int | None = None,
    home_day_form: float | None = None, away_day_form: float | None = None,
) -> dict:
    """Simulates from start_minute+1 to 90, given the state at start_minute.
    Defaults simulate a full match from kickoff.

    home_day_form / away_day_form: override the per-match day-form draw with
    a known value instead of sampling it -- mainly for tests that need a
    known ground truth to check the belief-update estimator against. None
    (the default) draws it from draw_day_form() as usual."""
    red_card_minute = {home.name: home_red_card_minute, away.name: away_red_card_minute}
    sent_off = {home.name: set(), away.name: set()}
    day_form = {
        home.name: home_day_form if home_day_form is not None else draw_day_form(rng),
        away.name: away_day_form if away_day_form is not None else draw_day_form(rng),
    }
    pressing = {home.name: False, away.name: False}
    shots = []
    red_card_events = []

    for minute in range(start_minute + 1, 91):
        for team, squad in ((home, home_squad), (away, away_squad)):
            if red_card_minute[team.name] is None and rng.random() < RED_CARD_CHANCE_PER_MINUTE:
                red_card_minute[team.name] = minute
                eligible = [p for p in squad if p.name not in sent_off[team.name]]
                carded = pick_player(rng, eligible, "card_share")
                sent_off[team.name].add(carded.name)
                red_card_events.append({"minute": minute, "team": team.name,
                                         "player": carded.name, "position": carded.position})

        for team in (home, away):
            if pressing[team.name]:
                if rng.random() < PRESSURE_EXIT_CHANCE_PER_MINUTE:
                    pressing[team.name] = False
            elif rng.random() < PRESSURE_ENTRY_CHANCE_PER_MINUTE:
                pressing[team.name] = True

        home_has_card = red_card_minute[home.name] is not None
        away_has_card = red_card_minute[away.name] is not None

        home_psi = day_form[home.name] * (PRESSURE_BOOST if pressing[home.name] else 1.0)
        away_psi = day_form[away.name] * (PRESSURE_BOOST if pressing[away.name] else 1.0)

        shot = _maybe_shot(rng, home, away, home_squad, sent_off[home.name], home_goals, away_goals,
                            minute, True, home_has_card, away_has_card, np.sqrt(home_psi))
        if shot is not None:
            shots.append({"minute": minute, "team": home.name,
                          "home_score": home_goals, "away_score": away_goals, **shot})
            if shot["is_goal"]:
                home_goals += 1

        shot = _maybe_shot(rng, away, home, away_squad, sent_off[away.name], away_goals, home_goals,
                            minute, False, away_has_card, home_has_card, np.sqrt(away_psi))
        if shot is not None:
            shots.append({"minute": minute, "team": away.name,
                          "home_score": home_goals, "away_score": away_goals, **shot})
            if shot["is_goal"]:
                away_goals += 1

    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "shots": shots,
        "red_card_events": red_card_events,
        "home_red_card_minute": red_card_minute[home.name],
        "away_red_card_minute": red_card_minute[away.name],
        "home_day_form": day_form[home.name],
        "away_day_form": day_form[away.name],
    }


def _maybe_shot(rng: np.random.Generator, team: Club, opponent: Club, squad: list[Player],
                 sent_off_names: set[str], own_score: int, opp_score: int, minute: int, is_home: bool,
                 team_has_red_card: bool, opp_has_red_card: bool, channel_multiplier: float) -> dict | None:
    rate = _shot_rate(team, opponent, own_score, opp_score, minute, is_home,
                       team_has_red_card, opp_has_red_card) * channel_multiplier
    if rng.random() >= rate:
        return None

    xg, is_penalty, is_big_chance = _draw_shot(rng)
    xg = float(np.clip(xg * channel_multiplier, 0.01, 0.99))
    eligible = [p for p in squad if p.name not in sent_off_names]
    shooter = pick_player(rng, eligible, "shot_share")
    effective_xg = float(np.clip(xg * shooter.finishing_skill, 0.01, 0.99))

    return {"xg": xg, "is_penalty": is_penalty, "is_big_chance": is_big_chance,
            "is_goal": rng.random() < effective_xg,
            "player": shooter.name, "position": shooter.position}


def _shot_rate(team: Club, opponent: Club, own_score: int, opp_score: int, minute: int,
                is_home: bool, team_has_red_card: bool, opp_has_red_card: bool) -> float:
    attack = team.attack
    defense = opponent.defense

    if own_score < opp_score:  # team is trailing
        attack *= team.comeback_factor          # team pushes harder
        defense *= opponent.protect_lead_factor  # opponent defends the lead

    if minute > 75:
        defense *= opponent.fatigue_factor  # opponent's legs tire late

    if team_has_red_card:
        attack *= team.ten_men_resilience
    if opp_has_red_card:
        defense *= opponent.ten_men_resilience

    rate = (BASE_SHOT_RATE_PER_90 / 90) * attack / defense
    if is_home:
        rate *= HOME_ADVANTAGE
    return rate


def _draw_shot(rng: np.random.Generator) -> tuple[float, bool, bool]:
    """Picks a shot type and its xG: penalty, big chance, or normal shot."""
    if rng.random() < PENALTY_SHOT_FRACTION:
        xg = float(np.clip(rng.normal(PENALTY_XG_MEAN, 0.03), 0.55, 0.9))
        return xg, True, False
    if rng.random() < BIG_CHANCE_FRACTION:
        xg = float(rng.uniform(*BIG_CHANCE_XG_RANGE))
        return xg, False, True
    xg = float(rng.uniform(*NORMAL_XG_RANGE))
    return xg, False, False
