"""Re-simulates a match forward from any point, many times in parallel, to
estimate win/draw/loss probability. Vectorized with numpy: every replay's
state is a slot in an array, and one minute is advanced for all replays
at once, rather than looping replay-by-replay.

For speed, this skips individual shot/player detail (not needed for a
final-score outcome) and instead uses a single combined "goal this minute"
probability per team, built from a shot-rate and shot-type mix of the same
SHAPE as the full match engine in match_engine.py, but with every number
in it estimated from the generated data (see
trait_estimation.estimate_league_constants / estimate_finishing_factor) --
never imported from match_engine.py's true constants.

Day form: match_engine.py gives each team a hidden per-match multiplier
psi (Gamma, mean 1) split as sqrt(psi) across BOTH the shot-rate and
shot-quality channels, so their product recovers psi exactly -- this
module never gets to see psi itself, only the REAL xG it produced. At any
checkpoint minute, this module's belief about a team's psi is the
Gamma-Poisson posterior mean given everything observed so far: prior
Gamma(k, 1/k) [mean 1], updated to Gamma(k + xg_observed, k + xg_expected)
where xg_expected is what the club-trait model (psi = 1) would have
predicted through that same minute. Taking total xG as the Poisson
"count" is what makes this match the generator's psi correctly (not
day_form itself, which only shows up per channel as sqrt(psi) and would
be a biased target to update on directly -- squaring back out after the
two channels compound would silently double-count the effect size).
k is learned from training data (trait_estimation.estimate_day_form_concentration),
not imported from match_engine.py's true DAY_FORM_SHAPE. Pressure spells
are a second, separate source of real match-to-match variance in
match_engine.py that this module deliberately does NOT try to track or
decompose -- it only ever asks "how much more/less xG has this team
produced than expected," which mixes day form and pressure together
indistinguishably. That's a known, reported limitation (see
scripts/estimate_traits.py's day-form report and the README) -- some of
what k learns is really pressure-spell noise, not true day-to-day
variation, which inflates the learned spread above the true day-form-only
spread.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.league import Club
from src.trait_estimation import match_real_states, sample_trait_draws, shrink_traits

# Every shot-rate/composition constant the fast engine needs is ESTIMATED from
# the generated data (see trait_estimation.estimate_league_constants) and
# saved to this file by scripts/estimate_traits.py -- never imported from
# src/match_engine.py (the true generator). Home advantage also lives here as
# a point estimate for convenience, but the uncertainty-aware path (team_draws)
# instead reads each club's own `home_advantage` field, which varies per
# sampled draw (see trait_estimation.sample_trait_draws /
# win_probability.clubs_from_traits).
_LEAGUE_CONSTANTS_PATH = Path(__file__).resolve().parent.parent / "data" / "model_league_constants.csv"


def _load_league_constants(path: Path = _LEAGUE_CONSTANTS_PATH) -> dict[str, float]:
    return pd.read_csv(path, index_col=0)["value"].to_dict()


_constants = _load_league_constants()
BASE_SHOT_RATE_PER_90 = _constants["base_shot_rate_per_90"]
PENALTY_SHOT_FRACTION = _constants["penalty_shot_fraction"]
BIG_CHANCE_FRACTION = _constants["big_chance_fraction"]
PENALTY_XG_MEAN = _constants["penalty_xg_mean"]
BIG_CHANCE_XG_MEAN = _constants["big_chance_xg_mean"]
NORMAL_XG_MEAN = _constants["normal_xg_mean"]
RED_CARD_CHANCE_PER_MINUTE = _constants["red_card_chance_per_minute"]
DAY_FORM_SHAPE = _constants["day_form_shape"]

# Expected xG per shot, blended across the three shot types by their estimated
# shares -- used only to convert an expected SHOT rate into an expected xG
# rate for the day-form belief update (the shot-rate/quality split itself
# still happens exactly as elsewhere; this is just the day-form bookkeeping).
MEAN_XG_PER_SHOT = (PENALTY_SHOT_FRACTION * PENALTY_XG_MEAN
                    + (1 - PENALTY_SHOT_FRACTION) * BIG_CHANCE_FRACTION * BIG_CHANCE_XG_MEAN
                    + (1 - PENALTY_SHOT_FRACTION) * (1 - BIG_CHANCE_FRACTION) * NORMAL_XG_MEAN)


def _shot_rate(team: Club, opponent: Club, own_score: np.ndarray, opp_score: np.ndarray,
               minute: int, is_home: bool, team_has_card: np.ndarray,
               opp_has_card: np.ndarray, day_form_multiplier: float = 1.0) -> np.ndarray:
    """Vectorized version of match_engine._shot_rate: same formula, but
    own_score/opp_score/*_has_card are arrays (one slot per replay), and
    home advantage comes from the HOME club's own estimated field (so it
    can vary per sampled trait draw) instead of a shared constant.
    day_form_multiplier is the rate-channel half of the day-form belief
    (see module docstring) -- 1.0 (no effect) unless the caller has
    computed one from real observed evidence."""
    trailing = own_score < opp_score
    attack = np.where(trailing, team.attack * team.comeback_factor, team.attack)
    defense = np.where(trailing, opponent.defense * opponent.protect_lead_factor, opponent.defense)

    if minute > 75:
        defense = defense * opponent.fatigue_factor

    attack = np.where(team_has_card, attack * team.ten_men_resilience, attack)
    defense = np.where(opp_has_card, defense * opponent.ten_men_resilience, defense)

    rate = (BASE_SHOT_RATE_PER_90 / 90) * attack / defense * day_form_multiplier
    if is_home:
        rate = rate * team.home_advantage
    return rate


def _goal_given_shot(rng: np.random.Generator, n: int, finishing_factor: float,
                      day_form_multiplier: float = 1.0) -> np.ndarray:
    """Whether a shot (already known to happen) is a goal: draws a shot
    type with the same mix as the full engine, using each type's mean xG,
    then applies the shooting club's estimated finishing_factor (the fast
    engine's club-level stand-in for the full engine's per-player
    finishing_skill) and the quality-channel half of the day-form belief
    (same day_form_multiplier passed to _shot_rate -- this mirrors
    match_engine.py applying sqrt(psi) to both channels)."""
    roll = rng.random(n)
    is_penalty = roll < PENALTY_SHOT_FRACTION
    is_big_chance = (~is_penalty) & (rng.random(n) < BIG_CHANCE_FRACTION)
    xg = np.where(is_penalty, PENALTY_XG_MEAN, np.where(is_big_chance, BIG_CHANCE_XG_MEAN, NORMAL_XG_MEAN))
    effective_xg = np.clip(xg * finishing_factor * day_form_multiplier, 0.01, 0.99)
    return rng.random(n) < effective_xg


def simulate_forward_snapshots(home: Club, away: Club, rng: np.random.Generator, start_minute: int,
                                home_goals: int, away_goals: int, home_has_card: bool, away_has_card: bool,
                                n_trials: int, snapshot_minutes: list[int],
                                home_day_form: float = 1.0, away_day_form: float = 1.0
                                ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Replays the rest of the match (start_minute+1 .. 90) n_trials times in
    parallel, in ONE pass, recording a copy of every replay's score at each
    requested minute. Used when later checkpoints should reflect a single
    coherent imagined continuation (e.g. a counterfactual "what if" branch),
    rather than independently re-simulating from a real, known state at each
    checkpoint (that's what win_probability_curve does for the real match).

    home_day_form / away_day_form: the Bayesian posterior-mean day-form
    multiplier for the REST of the match, held fixed for every trial and
    every remaining minute (a point estimate, not a fresh per-trial draw --
    the whole point is to react to what's already been observed, not to
    simulate hypothetical future day-form swings). 1.0 (the default)
    reproduces the pre-day-form behavior exactly, used for the "without
    the update" side of the validation ablation."""
    home_score = np.full(n_trials, home_goals, dtype=np.int32)
    away_score = np.full(n_trials, away_goals, dtype=np.int32)
    home_card = np.full(n_trials, home_has_card)
    away_card = np.full(n_trials, away_has_card)

    snapshot_set = set(snapshot_minutes)
    snapshots = {}
    if start_minute in snapshot_set:
        snapshots[start_minute] = (home_score.copy(), away_score.copy())

    for minute in range(start_minute + 1, 91):
        home_card |= rng.random(n_trials) < RED_CARD_CHANCE_PER_MINUTE
        away_card |= rng.random(n_trials) < RED_CARD_CHANCE_PER_MINUTE

        home_rate = _shot_rate(home, away, home_score, away_score, minute, True, home_card, away_card,
                                home_day_form)
        away_rate = _shot_rate(away, home, away_score, home_score, minute, False, away_card, home_card,
                                away_day_form)

        home_shot = rng.random(n_trials) < home_rate
        away_shot = rng.random(n_trials) < away_rate

        home_score += (home_shot & _goal_given_shot(rng, n_trials, home.finishing_factor,
                                                      home_day_form)).astype(np.int32)
        away_score += (away_shot & _goal_given_shot(rng, n_trials, away.finishing_factor,
                                                      away_day_form)).astype(np.int32)

        if minute in snapshot_set:
            snapshots[minute] = (home_score.copy(), away_score.copy())

    return snapshots


def simulate_forward(home: Club, away: Club, rng: np.random.Generator, start_minute: int,
                      home_goals: int, away_goals: int, home_has_card: bool, away_has_card: bool,
                      n_trials: int, home_day_form: float = 1.0, away_day_form: float = 1.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Replays the rest of the match (start_minute+1 .. 90) n_trials times
    in parallel. Returns (final_home_score, final_away_score) arrays."""
    return simulate_forward_snapshots(home, away, rng, start_minute, home_goals, away_goals,
                                       home_has_card, away_has_card, n_trials, [90],
                                       home_day_form, away_day_form)[90]


def win_probabilities(final_home: np.ndarray, final_away: np.ndarray) -> dict:
    """Home/draw/away win shares, with a standard-error band on each,
    from a finite number of Monte Carlo replays."""
    n = len(final_home)
    p_home = (final_home > final_away).mean()
    p_draw = (final_home == final_away).mean()
    p_away = (final_home < final_away).mean()
    se = lambda p: np.sqrt(p * (1 - p) / n)
    return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "se_home": se(p_home), "se_draw": se(p_draw), "se_away": se(p_away)}


def real_state_at(real_states: pd.DataFrame, final_score: tuple[int, int], minute: int) -> tuple:
    """The real match's actual (score, red-card) state as of the END of
    `minute`. real_states: one row per minute 1..90 as produced by
    trait_estimation._match_minute_states (the state used for that minute's
    own rate calc, i.e. BEFORE that minute's events). final_score: the
    match's true final score (real_states only covers pre-minute-90 state,
    not the full-time result)."""
    last_row = real_states.iloc[-1]  # minute-90 row: state entering minute 90
    if minute == 0:
        return 0, 0, False, False
    if minute == 90:
        return final_score[0], final_score[1], bool(last_row.home_card), bool(last_row.away_card)
    next_row = real_states[real_states.minute == minute + 1].iloc[0]
    return (int(next_row.home_score), int(next_row.away_score),
            bool(next_row.home_card), bool(next_row.away_card))


def _expected_cumulative_xg(club: Club, opponent: Club, real_states: pd.DataFrame, is_home: bool) -> np.ndarray:
    """Expected (day-form-neutral, psi=1) cumulative xG for `club`, through
    each minute 0..90 inclusive -- arr[m] = expected total through minute m.
    This is the Gamma-Poisson exposure (Lambda) day_form_multiplier_curves
    conditions against; real_states supplies the real game state
    (score/cards) at each minute, same as everywhere else in this module."""
    arr = np.zeros(91)
    total = 0.0
    for _, row in real_states.iterrows():
        minute = int(row.minute)
        own_score = np.array([row.home_score if is_home else row.away_score])
        opp_score = np.array([row.away_score if is_home else row.home_score])
        own_card = np.array([bool(row.home_card if is_home else row.away_card)])
        opp_card = np.array([bool(row.away_card if is_home else row.home_card)])
        rate = _shot_rate(club, opponent, own_score, opp_score, minute, is_home, own_card, opp_card)[0]
        total += rate * MEAN_XG_PER_SHOT
        arr[minute] = total
    return arr


def _observed_cumulative_xg_through(real_states: pd.DataFrame, minute: int) -> tuple[float, float]:
    """Real, observed cumulative xG for each side through `minute`
    inclusive -- same "through minute m" convention as _expected_cumulative_xg,
    reusing real_state_at's pattern of reading the next row (or, past
    minute 90, there's nothing left to simulate anyway so the exact value
    stops mattering)."""
    if minute == 0:
        return 0.0, 0.0
    if minute >= 90:
        last = real_states.iloc[-1]
        return float(last.home_xg_cumulative), float(last.away_xg_cumulative)
    next_row = real_states[real_states.minute == minute + 1].iloc[0]
    return float(next_row.home_xg_cumulative), float(next_row.away_xg_cumulative)


def day_form_multiplier_curves(home: Club, away: Club, real_states: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For every minute 0..90, the Bayesian posterior-mean day-form
    multiplier for home and away (see module docstring), given everything
    observed in the REAL match up to that minute. Index 0 is exactly 1.0
    for both sides -- no evidence yet, so the belief hasn't moved from the
    neutral prior (see tests/test_day_form.py's
    test_day_form_multiplier_is_neutral_with_no_evidence)."""
    home_expected = _expected_cumulative_xg(home, away, real_states, True)
    away_expected = _expected_cumulative_xg(away, home, real_states, False)

    home_mult, away_mult = np.ones(91), np.ones(91)
    for minute in range(91):
        home_obs, away_obs = _observed_cumulative_xg_through(real_states, minute)
        home_mult[minute] = np.sqrt((DAY_FORM_SHAPE + home_obs) / (DAY_FORM_SHAPE + home_expected[minute]))
        away_mult[minute] = np.sqrt((DAY_FORM_SHAPE + away_obs) / (DAY_FORM_SHAPE + away_expected[minute]))
    return home_mult, away_mult


def win_probability_curve(home: Club, away: Club, real_states: pd.DataFrame, final_score: tuple[int, int],
                           seed: int, n_trials: int, checkpoints: list[int],
                           use_day_form: bool = True) -> pd.DataFrame:
    """Win probability at each checkpoint minute, re-simulating forward from
    the REAL match's actual state at that minute each time (independent
    per checkpoint — appropriate since the real continuation is known).
    Reflects simulation noise only; see win_probability_curve_with_uncertainty
    for a version that also reflects trait-estimation uncertainty.

    Uses the same `seed` to start every checkpoint's simulation (common
    random numbers): adjacent checkpoints then share most of their random
    draws, so the curve only moves meaningfully where the real state
    actually changed, instead of jittering from independent Monte Carlo
    noise between checkpoints that are simulating almost the same thing.

    use_day_form: whether to react to the real match's observed shots/xG
    so far via the Bayesian day-form belief (see module docstring).
    False reproduces the pre-day-form curve exactly -- used for the "without
    the update" side of the validation ablation in scripts/calibration.py."""
    if use_day_form:
        home_day_form_curve, away_day_form_curve = day_form_multiplier_curves(home, away, real_states)

    rows = []
    for minute in checkpoints:
        h_score, a_score, h_card, a_card = real_state_at(real_states, final_score, minute)
        home_df = home_day_form_curve[minute] if use_day_form else 1.0
        away_df = away_day_form_curve[minute] if use_day_form else 1.0
        rng = np.random.default_rng(seed)
        final_home, final_away = simulate_forward(home, away, rng, minute, h_score, a_score,
                                                    h_card, a_card, n_trials, home_df, away_df)
        probs = win_probabilities(final_home, final_away)
        rows.append({"minute": minute, "home_score": h_score, "away_score": a_score, **probs})

    return pd.DataFrame(rows)


def _pool_batches(batch_probs: list[dict]) -> dict:
    """Combines per-batch win_probabilities() dicts (one per sampled trait
    draw) into an overall mean +/- standard error of that mean. Because
    each batch used a different plausible trait draw, the spread across
    batches captures parameter uncertainty on top of each batch's own
    simulation noise (law of total variance) -- wider where the trait
    estimates themselves were less certain."""
    batch_df = pd.DataFrame(batch_probs)
    n_batches = len(batch_probs)
    row = {}
    for outcome in ("home", "draw", "away"):
        col = f"p_{outcome}"
        row[col] = batch_df[col].mean()
        row[f"se_{outcome}"] = batch_df[col].std(ddof=1) / np.sqrt(n_batches)
    return row


def win_probability_curve_with_uncertainty(team_draws: list[dict[str, Club]], home_name: str, away_name: str,
                                            real_states: pd.DataFrame, final_score: tuple[int, int],
                                            seed: int, trials_per_batch: int,
                                            checkpoints: list[int], use_day_form: bool = True) -> pd.DataFrame:
    """Like win_probability_curve, but each of the len(team_draws) batches
    uses a different sampled set of club traits (see
    trait_estimation.sample_trait_draws), so the resulting bands reflect
    both simulation noise and trait-estimation uncertainty. The day-form
    belief (see module docstring) is computed separately per draw, since
    each draw's own attack/defense gives a slightly different "expected"
    baseline to compare the real observed xG against.

    Common random numbers: batch i always starts from seed + i, regardless
    of which checkpoint it's simulating, so adjacent checkpoints are highly
    correlated and the curve doesn't jitter between them (see
    win_probability_curve for the full rationale)."""
    day_form_curves = None
    if use_day_form:
        day_form_curves = [day_form_multiplier_curves(draw[home_name], draw[away_name], real_states)
                            for draw in team_draws]

    rows = []
    for minute in checkpoints:
        h_score, a_score, h_card, a_card = real_state_at(real_states, final_score, minute)
        batch_probs = []
        for i, draw in enumerate(team_draws):
            rng = np.random.default_rng(seed + i)
            home_df, away_df = (day_form_curves[i][0][minute], day_form_curves[i][1][minute]) \
                if use_day_form else (1.0, 1.0)
            final_home, final_away = simulate_forward(draw[home_name], draw[away_name], rng, minute,
                                                        h_score, a_score, h_card, a_card, trials_per_batch,
                                                        home_df, away_df)
            batch_probs.append(win_probabilities(final_home, final_away))
        rows.append({"minute": minute, "home_score": h_score, "away_score": a_score,
                     **_pool_batches(batch_probs)})

    return pd.DataFrame(rows)


def counterfactual_fork_probabilities(team_draws: list[dict[str, Club]], home_name: str, away_name: str,
                                       seed: int, fork_minute: int, home_goals: int, away_goals: int,
                                       home_has_card: bool, away_has_card: bool,
                                       trials_per_batch: int, real_states: pd.DataFrame | None = None,
                                       use_day_form: bool = True) -> dict:
    """The one well-defined number after a hypothetical change: probability
    of each outcome by full time, simulated forward from the fork state,
    pooled across trait draws. There is no single "counterfactual
    timeline" beyond this -- only a distribution over what might happen --
    so this is deliberately a single forecast, not a minute-by-minute
    curve (see counterfactual_continuation_curve).

    real_states: the REAL match's observed state (not the hypothetical
    forked one) -- day form reflects how the team has actually been
    playing, which a hypothetical change to one event doesn't alter.
    Required (and used) only when use_day_form is True."""
    batch_probs = []
    for i, draw in enumerate(team_draws):
        rng = np.random.default_rng(seed + i)
        if use_day_form and real_states is not None:
            home_curve, away_curve = day_form_multiplier_curves(draw[home_name], draw[away_name], real_states)
            home_df, away_df = home_curve[fork_minute], away_curve[fork_minute]
        else:
            home_df, away_df = 1.0, 1.0
        final_home, final_away = simulate_forward(draw[home_name], draw[away_name], rng, fork_minute,
                                                    home_goals, away_goals, home_has_card, away_has_card,
                                                    trials_per_batch, home_df, away_df)
        batch_probs.append(win_probabilities(final_home, final_away))
    return _pool_batches(batch_probs)


def counterfactual_continuation_curve(team_draws: list[dict[str, Club]], home_name: str, away_name: str,
                                       seed: int, fork_minute: int, home_goals: int,
                                       away_goals: int, home_has_card: bool, away_has_card: bool,
                                       trials_per_batch: int, checkpoints: list[int],
                                       real_states: pd.DataFrame | None = None,
                                       use_day_form: bool = True) -> pd.DataFrame:
    """Win probability for every checkpoint >= fork_minute, as an honest
    FLAT line at the single fork forecast (see
    counterfactual_fork_probabilities), not a fabricated trajectory.

    An earlier version traced one continuous simulated path per trait draw
    and computed win_probabilities() from the CURRENT score at each
    snapshot minute -- but that measures "share of replays currently level
    or ahead at minute t", not "probability of winning by minute 90 given
    this state", and is a different (and wrong) quantity: right after a
    fork that ties the score, it reads as ~0% "ahead" and climbs only as
    the single-path noise happens to separate, even though the real
    full-time win probability jumps immediately. There is no single
    well-defined counterfactual trajectory to trace minute by minute after
    a fork -- only a distribution over what might happen -- so the honest
    summary is one forecast, held flat, with its uncertainty band."""
    probs = counterfactual_fork_probabilities(team_draws, home_name, away_name, seed, fork_minute,
                                               home_goals, away_goals, home_has_card, away_has_card,
                                               trials_per_batch, real_states, use_day_form)
    return pd.DataFrame([{"minute": m, **probs} for m in checkpoints])


def clubs_from_traits(traits: pd.DataFrame) -> dict[str, Club]:
    """Converts a traits DataFrame (indexed by team) into Club objects.
    home_advantage/finishing_factor default to 1.0 (no effect) when the
    traits table doesn't carry them -- e.g. scripts/calibration.py's
    out-of-sample refit doesn't estimate finishing_factor separately."""
    return {team: Club(name=team, attack=row.attack, defense=row.defense,
                        comeback_factor=row.comeback_factor, protect_lead_factor=row.protect_lead_factor,
                        ten_men_resilience=row.ten_men_resilience, fatigue_factor=row.fatigue_factor,
                        home_advantage=row.get("home_advantage", 1.0),
                        finishing_factor=row.get("finishing_factor", 1.0))
            for team, row in traits.iterrows()}


def compute_power_rating(clubs: dict[str, Club]) -> pd.DataFrame:
    """An explicit club rating in "expected goal difference per match"
    units: how many more (or fewer) goals this club would be expected to
    score minus concede per match against a hypothetical AVERAGE opponent
    (attack = defense = 1.0) on a neutral ground (no home advantage) --
    read directly off the estimated attack/defense, not a separate model.
    Sorted strongest to weakest."""
    avg_goals_per_match = BASE_SHOT_RATE_PER_90 * MEAN_XG_PER_SHOT
    rows = []
    for team, club in clubs.items():
        expected_goals_for = avg_goals_per_match * club.attack
        expected_goals_against = avg_goals_per_match / club.defense
        rows.append({"team": team, "expected_goal_difference": expected_goals_for - expected_goals_against,
                     "expected_goals_for": expected_goals_for, "expected_goals_against": expected_goals_against})
    return pd.DataFrame(rows).sort_values("expected_goal_difference", ascending=False).reset_index(drop=True)


def sample_team_draws(params: pd.Series, cov: pd.DataFrame, teams: list[str], exposure: pd.DataFrame,
                       shrink_cols: list[str], n_draws: int, rng: np.random.Generator,
                       finishing_factor: pd.Series | None = None) -> list[dict[str, Club]]:
    """Samples n_draws alternate leagues from the GLM's uncertainty,
    applying the same selective shrinkage used for the point estimate so
    the draws are centered consistently with it. home_advantage comes along
    for free (one of the GLM's own coefficients, so it varies naturally
    per draw); finishing_factor isn't part of the GLM at all (see
    trait_estimation.estimate_finishing_factor), so the same point
    estimate is attached to every draw if supplied."""
    raw_draws = sample_trait_draws(params, cov, teams, n_draws, rng)
    shrunk_draws = [shrink_traits(d, exposure, shrink_cols) for d in raw_draws]
    if finishing_factor is not None:
        for d in shrunk_draws:
            d["finishing_factor"] = finishing_factor
    return [clubs_from_traits(d) for d in shrunk_draws]


def rank_matches_by_drama(matches: pd.DataFrame, shots: pd.DataFrame, red_cards: pd.DataFrame,
                           clubs: dict[str, Club], seed: int, n_trials: int = 150,
                           checkpoints: list[int] | None = None) -> pd.DataFrame:
    """Coarse (fast) home-win-probability swing for every match, using
    point-estimate traits, sorted most to least dramatic. Used both to
    pick the single most dramatic match and to build a shortlist of
    several interesting ones to precompute for instant loading."""
    checkpoints = checkpoints if checkpoints is not None else list(range(0, 91, 10))
    rows = []
    for _, match in matches.iterrows():
        states = match_real_states(match, shots, red_cards)
        curve = win_probability_curve(clubs[match.home_team], clubs[match.away_team], states,
                                       (match.home_goals, match.away_goals), seed, n_trials, checkpoints)
        swing = curve.p_home.max() - curve.p_home.min()
        rows.append({"match_id": match.match_id, "swing": swing})
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def find_most_dramatic_match(matches: pd.DataFrame, shots: pd.DataFrame, red_cards: pd.DataFrame,
                              clubs: dict[str, Club], seed: int,
                              n_trials: int = 150, checkpoints: list[int] | None = None) -> tuple[pd.Series, float]:
    """Screens every match for the largest home-win-probability swing. Returns (match_row, swing)."""
    ranked = rank_matches_by_drama(matches, shots, red_cards, clubs, seed, n_trials, checkpoints)
    top = ranked.iloc[0]
    best_match = matches[matches.match_id == top.match_id].iloc[0]
    return best_match, float(top.swing)


# Priority for attributing a turning point when multiple events share a minute:
# state-changing events (goals, red cards) actually move the simulated win
# probability; missed shots don't change the score or cards, so they can
# never be what drove a swing, even if one happened in the same minute.
_EVENT_PRIORITY = {"goal": 0, "red_card": 1, "big_chance": 2, "missed_penalty": 2}


def events_at_minute(minute: int, match: pd.Series, shots: pd.DataFrame, red_cards: pd.DataFrame) -> list[dict]:
    """All notable events at this exact minute, ranked by _EVENT_PRIORITY so
    the first one is whichever actually could have driven a win-probability
    swing (a goal or red card), not a merely-coincidental missed shot."""
    match_shots = shots[shots.match_id == match.match_id]
    match_cards = red_cards[red_cards.match_id == match.match_id]

    events = []
    for _, r in match_shots[match_shots.minute == minute].iterrows():
        if r.is_goal:
            kind = "goal"
        elif r.is_penalty:
            kind = "missed_penalty"
        elif r.is_big_chance:
            kind = "big_chance"
        else:
            continue
        events.append({"kind": kind, "team": r.team, "player": r.player, "is_penalty": bool(r.is_penalty)})

    for _, r in match_cards[match_cards.minute == minute].iterrows():
        events.append({"kind": "red_card", "team": r.team, "player": r.player, "is_penalty": False})

    events.sort(key=lambda e: _EVENT_PRIORITY[e["kind"]])
    return events


def event_minutes(match: pd.Series, shots: pd.DataFrame, red_cards: pd.DataFrame) -> set[int]:
    """Minutes with a real, state-changing event (a goal or a red card) --
    the only minutes that can legitimately be turning points. Excludes
    missed shots, which never change the simulated score/card state."""
    match_shots = shots[(shots.match_id == match.match_id) & shots.is_goal]
    match_cards = red_cards[red_cards.match_id == match.match_id]
    return set(match_shots.minute.tolist()) | set(match_cards.minute.tolist())


def rank_turning_points(curve: pd.DataFrame, match: pd.Series, shots: pd.DataFrame,
                         red_cards: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Ranks REAL events (goals and red cards only) by total movement across
    all three outcome probabilities -- never a minute with no event, which
    would just be Monte Carlo noise between checkpoints, not a turning
    point. When more than one event shares a minute, the state-changing one
    (a goal or red card) is attributed as the primary driver; any other
    event at that minute is kept as secondary context."""
    candidate_minutes = event_minutes(match, shots, red_cards)
    c = curve.copy()
    c["swing"] = (c.p_home.diff().abs() + c.p_draw.diff().abs() + c.p_away.diff().abs()) / 2
    c = c[c.minute.isin(candidate_minutes)]
    top = c.nlargest(min(top_n, len(c)), "swing").sort_values("minute").reset_index(drop=True)

    columns = ["minute", "swing", "p_home", "p_draw", "p_away", "event_kind", "event_team",
               "event_player", "is_penalty", "secondary_events", "headline_outcome",
               "headline_before", "headline_after"]
    if len(top) == 0:
        # No goals or red cards at all (e.g. a goalless, card-free draw) -- there's
        # nothing that could legitimately be a turning point. Keep the columns so
        # callers can rely on them existing even when there are zero rows.
        return pd.DataFrame(columns=columns)

    indexed = curve.set_index("minute")
    rows = []
    for _, row in top.iterrows():
        minute = int(row.minute)
        events = events_at_minute(minute, match, shots, red_cards)
        primary, secondary = events[0], events[1:]

        before = indexed.loc[minute - 1]
        deltas = {outcome: row[f"p_{outcome}"] - before[f"p_{outcome}"] for outcome in ("home", "draw", "away")}
        headline_outcome = max(deltas, key=lambda o: abs(deltas[o]))

        rows.append({
            "minute": minute, "swing": row.swing,
            "p_home": row.p_home, "p_draw": row.p_draw, "p_away": row.p_away,
            "event_kind": primary["kind"], "event_team": primary["team"], "event_player": primary["player"],
            "is_penalty": primary["is_penalty"], "secondary_events": secondary,
            "headline_outcome": headline_outcome,
            "headline_before": before[f"p_{headline_outcome}"],
            "headline_after": row[f"p_{headline_outcome}"],
        })

    return pd.DataFrame(rows)[columns]
