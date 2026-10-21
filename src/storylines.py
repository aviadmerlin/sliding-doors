"""Turns the generator's between-season trait jumps (src/league.py's
drift_traits) into short plain-language storylines for the app's club
profile card. Text only -- reads already-generated data
(data/trait_jumps.csv), no model or estimation logic.

For all six traits, a BIGGER value is the improvement (attack/defense/
comeback/lead-protection are boosts that help; ten_men_resilience/
fatigue_factor are <=1.0 penalty multipliers where closer to 1.0 means
less of a penalty) -- so delta > 0 always means "got better" and no
per-trait sign-flipping is needed."""

import pandas as pd

_UP_PHRASES = {
    "attack": "an attacking signing paid off",
    "defense": "defense improved sharply",
    "comeback_factor": "started finding comebacks from losing positions",
    "protect_lead_factor": "got much better at protecting a lead",
    "ten_men_resilience": "coped much better playing with ten men",
    "fatigue_factor": "held up much better physically late in games",
}
_DOWN_PHRASES = {
    "attack": "lost their cutting edge up front",
    "defense": "defense fell apart",
    "comeback_factor": "stopped finding a way back from behind",
    "protect_lead_factor": "started throwing away leads",
    "ten_men_resilience": "really struggled whenever reduced to ten men",
    "fatigue_factor": "started fading badly in the last 15 minutes",
}

_LEAD_INS = ["New coach appointed", "Squad overhaul in the off-season", "New-look team this season"]


def club_storylines(team: str, trait_jumps: pd.DataFrame, top_n_traits: int = 1) -> list[str]:
    """One short sentence per jump season for this club, e.g. 'Season 4:
    new coach appointed, defense improved sharply.'"""
    team_jumps = trait_jumps[trait_jumps.team == team]
    lines = []
    for season, group in team_jumps.groupby("season"):
        deltas = dict(zip(group.trait, group.delta))
        headline = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n_traits]
        phrases = [(_UP_PHRASES if delta > 0 else _DOWN_PHRASES)[trait] for trait, delta in headline]
        lead_in = _LEAD_INS[int(season) % len(_LEAD_INS)]
        lines.append(f"Season {int(season)}: {lead_in.lower()}, {', '.join(phrases)}.")
    return sorted(lines)
