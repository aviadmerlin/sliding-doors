"""Tools wrapping the existing engine for the AI agent (src/agent.py) --
each one a thin, pure wrapper returning plain dicts, so results are
always JSON-serializable (and directly inspectable in tests) with no
LLM-specific code anywhere in src/. All four mirror functionality the
Streamlit app (app.py) already exposes -- same underlying src/
functions, just packaged for tool-calling instead of rendering.
"""

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import Field

from src.counterfactual import CHANGE_TYPES, Counterfactual, apply_counterfactual
from src.storylines import club_profile
from src.trait_estimation import match_real_states
from src.win_probability import (clubs_from_traits, list_notable_events, rank_turning_points,
                                  win_probability_curve_with_uncertainty)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Must match app.py / scripts/precompute_featured_matches.py so on-demand
# curves are directly comparable to the precomputed featured ones.
TRIALS_PER_DRAW = 200
CHECKPOINTS = list(range(0, 91))


class MatchContext:
    """Loads everything needed to answer questions about ONE match, once,
    and exposes it as tool methods an LLM agent can call (see .tools()).
    Construct one fresh instance per match_id -- cheap (CSV reads + the
    already-fitted model from disk, no refitting)."""

    def __init__(self, match_id: int):
        self.match_id = match_id
        self.matches = pd.read_csv(DATA_DIR / "matches.csv")
        self.shots = pd.read_csv(DATA_DIR / "shots.csv")
        self.red_cards = pd.read_csv(DATA_DIR / "red_cards.csv")
        self.estimated_traits = pd.read_csv(DATA_DIR / "estimated_traits.csv").set_index("team")

        matching = self.matches[self.matches.match_id == match_id]
        if len(matching) == 0:
            raise ValueError(f"No match with match_id={match_id}")
        self.match = matching.iloc[0]

        team_draws_df = pd.read_csv(DATA_DIR / "model_team_draws.csv")
        self.team_draws = [clubs_from_traits(g.set_index("team")) for _, g in team_draws_df.groupby("draw")]

        featured_meta = json.loads((DATA_DIR / "featured_matches.json").read_text())
        self.featured_ids = set(featured_meta["featured_match_ids"])
        self.featured_curves = pd.read_csv(DATA_DIR / "featured_curves.csv")
        self.featured_turning_points = pd.read_csv(DATA_DIR / "featured_turning_points.csv")

        self._curve_cache: pd.DataFrame | None = None

    def _curve(self) -> pd.DataFrame:
        """The match's win-probability curve -- precomputed if featured,
        else simulated once here and cached for the rest of this context's
        life (so repeated tool calls within one conversation don't re-simulate)."""
        if self._curve_cache is not None:
            return self._curve_cache
        if self.match_id in self.featured_ids:
            self._curve_cache = (self.featured_curves[self.featured_curves.match_id == self.match_id]
                                  .drop(columns="match_id").reset_index(drop=True))
            return self._curve_cache
        states = match_real_states(self.match, self.shots, self.red_cards)
        self._curve_cache = win_probability_curve_with_uncertainty(
            self.team_draws, self.match.home_team, self.match.away_team, states,
            (int(self.match.home_goals), int(self.match.away_goals)),
            1000 + self.match_id, TRIALS_PER_DRAW, CHECKPOINTS)
        return self._curve_cache

    def _turning_points(self) -> pd.DataFrame:
        if self.match_id in self.featured_ids:
            return (self.featured_turning_points[self.featured_turning_points.match_id == self.match_id]
                    .drop(columns="match_id").reset_index(drop=True))
        return rank_turning_points(self._curve(), self.match, self.shots, self.red_cards, top_n=3)

    # --- Tools exposed to the Analyst agent ---

    def list_match_events(self) -> list[dict]:
        """Lists every notable event in this match (goals, missed penalties, missed big
        chances, red cards), in chronological order with minute, team, player, and a short
        description. Use this first to find the exact minute/team/player of an event a
        question refers to -- never guess a minute or player name."""
        events = list_notable_events(self.match, self.shots, self.red_cards)
        return events.to_dict(orient="records")

    def get_win_probability(
        self,
        minute: Annotated[int, Field(description="Minute of the match (0-90) to get the "
                                                   "win probability at.", ge=0, le=90)],
    ) -> dict:
        """Gets the model's win/draw/loss probability for this match at a specific minute,
        based on the real game state up to that point."""
        curve = self._curve()
        row = curve[curve.minute == minute]
        if len(row) == 0:
            raise ValueError(f"No win-probability data at minute {minute} (valid range: 0-90)")
        row = row.iloc[0]
        return {
            "minute": minute, "home_team": self.match.home_team, "away_team": self.match.away_team,
            "p_home": round(float(row.p_home), 4), "p_draw": round(float(row.p_draw), 4),
            "p_away": round(float(row.p_away), 4),
        }

    def run_counterfactual(
        self,
        minute: Annotated[int, Field(description="The minute of the REAL event being changed "
                                                   "(get this from list_match_events, never guessed).")],
        change: Annotated[str, Field(description=f"One of: {', '.join(CHANGE_TYPES)}. No other "
                                                   f"kind of change can be simulated.")],
        team: Annotated[str | None, Field(description="Which team's event -- only needed if "
                                                        "more than one qualifying event happened "
                                                        "at that exact minute.")] = None,
    ) -> dict:
        """Re-simulates the match with ONE real event changed, returning a fair
        forecast-vs-forecast before/after win-probability comparison at the moment of the
        change. Supported changes: shot_becomes_goal (a missed shot/penalty scores instead),
        goal_cancelled (a scored goal is wiped out), red_card_undone (a red card is removed).
        No other kind of change (e.g. a tactical or personnel change) can be simulated."""
        if change not in CHANGE_TYPES:
            raise ValueError(f"Unsupported change {change!r}; must be one of {CHANGE_TYPES}")
        cf = Counterfactual(match_id=self.match_id, minute=minute, change=change, team=team)
        seed = 2000 + self.match_id * 97 + minute
        result = apply_counterfactual(cf, self.matches, self.shots, self.red_cards, self.team_draws,
                                       seed, TRIALS_PER_DRAW, CHECKPOINTS, actual_curve=self._curve())

        # Which of home/draw/away moved the most -- the one number the answer
        # should lead with, so the Narrator never has to guess which of the
        # three outcomes is the headline (same rule the UI's turning-point
        # cards use: biggest total movement).
        deltas = {o: result.fork_after[o] - result.fork_before[o] for o in ("home", "draw", "away")}
        headline_outcome = max(deltas, key=lambda o: abs(deltas[o]))
        headline_team = {"home": self.match.home_team, "draw": "a draw",
                         "away": self.match.away_team}[headline_outcome]

        return {
            "description": result.description,
            "fork_minute": result.fork_minute,
            "forecast_before_change": {k: round(v, 4) for k, v in result.fork_before.items()},
            "forecast_after_change": {k: round(v, 4) for k, v in result.fork_after.items()},
            "headline_team": headline_team,
            "headline_before": round(result.fork_before[headline_outcome], 4),
            "headline_after": round(result.fork_after[headline_outcome], 4),
            "real_final_score": {"home": result.real_final_score[0], "away": result.real_final_score[1]},
            "real_outcome": result.real_outcome,
        }

    def get_club_profile(
        self,
        team: Annotated[str, Field(description="Exact club name, e.g. 'Silverlake FC'.")],
    ) -> dict:
        """Gets plain-language notable traits for a club (e.g. strong attack, weak at
        protecting a lead), with a confidence note on which estimates are solid vs. noisier."""
        if team not in self.estimated_traits.index:
            raise ValueError(f"Unknown club {team!r}; must be an exact club name, e.g. "
                              f"{self.match.home_team!r} or {self.match.away_team!r}")
        return {"team": team, "notable_traits": club_profile(team, self.estimated_traits)}

    def tools(self) -> list:
        """The question-answering tools for the Analyst agent (see src/agent.py)."""
        return [self.list_match_events, self.get_win_probability, self.run_counterfactual,
                self.get_club_profile]

    # --- Recap data (not an LLM tool -- the match is already fixed, nothing to route) ---

    def get_match_recap_data(self) -> dict:
        """Final result and the match's biggest turning points (minute, event, before/after
        win probability) -- everything the Narrator needs to write a short match recap."""
        turning_points = self._turning_points()
        tp_records = [{
            "minute": int(tp.minute), "event_kind": tp.event_kind, "event_team": tp.event_team,
            "event_player": tp.event_player, "headline_outcome": tp.headline_outcome,
            "before": round(float(tp.headline_before), 4), "after": round(float(tp.headline_after), 4),
        } for _, tp in turning_points.iterrows()]
        return {
            "home_team": self.match.home_team, "away_team": self.match.away_team,
            "home_goals": int(self.match.home_goals), "away_goals": int(self.match.away_goals),
            "turning_points": tp_records,
        }
