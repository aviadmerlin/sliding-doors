"""Category 1: data integrity. Checks the generated CSVs themselves are
internally consistent, with no reference to the estimation or simulation
model -- pure bookkeeping checks on matches.csv / shots.csv / red_cards.csv
/ players.csv."""

import pandas as pd


def test_match_score_equals_goals_in_shots(matches: pd.DataFrame, shots: pd.DataFrame) -> None:
    goal_counts = shots[shots.is_goal].groupby(["match_id", "team"]).size()
    mismatches = []
    for _, m in matches.iterrows():
        home_recorded = int(goal_counts.get((m.match_id, m.home_team), 0))
        away_recorded = int(goal_counts.get((m.match_id, m.away_team), 0))
        if home_recorded != m.home_goals or away_recorded != m.away_goals:
            mismatches.append((m.match_id, "home", m.home_goals, home_recorded))
            mismatches.append((m.match_id, "away", m.away_goals, away_recorded))
    assert not mismatches, (
        f"{len(mismatches)} team-scores disagree with shots.csv goal counts "
        f"(match_id, side, matches.csv value, shots.csv count): {mismatches[:10]}")


def test_goals_csv_matches_shots_is_goal_filter(shots: pd.DataFrame, goals_csv: pd.DataFrame) -> None:
    """data/goals.csv should be exactly the is_goal rows of shots.csv (the
    precompute/generation pipeline derives it that way) -- catches the two
    files drifting apart after a regeneration."""
    expected = shots[shots.is_goal].drop(columns=["is_goal"]).reset_index(drop=True)
    actual = goals_csv.reset_index(drop=True)
    assert list(expected.columns) == list(actual.columns), (
        f"column mismatch: shots-derived={list(expected.columns)} vs goals.csv={list(actual.columns)}")
    assert len(expected) == len(actual), f"{len(expected)} is_goal rows in shots.csv vs {len(actual)} in goals.csv"
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False)


def test_shots_belong_to_correct_team_roster(shots: pd.DataFrame, players: pd.DataFrame) -> None:
    valid_pairs = set(zip(players.team, players.player))
    mask = [pair in valid_pairs for pair in zip(shots.team, shots.player)]
    bad = shots[~pd.Series(mask, index=shots.index)]
    assert bad.empty, f"{len(bad)} shots by a (team, player) pair not on that club's roster:\n{bad.head(10)}"


def test_red_cards_belong_to_correct_team_roster(red_cards: pd.DataFrame, players: pd.DataFrame) -> None:
    valid_pairs = set(zip(players.team, players.player))
    mask = [pair in valid_pairs for pair in zip(red_cards.team, red_cards.player)]
    bad = red_cards[~pd.Series(mask, index=red_cards.index)]
    assert bad.empty, f"{len(bad)} red cards by a (team, player) pair not on that club's roster:\n{bad.head(10)}"


def test_shot_minutes_within_1_to_90(shots: pd.DataFrame) -> None:
    bad = shots[(shots.minute < 1) | (shots.minute > 90)]
    assert bad.empty, f"{len(bad)} shots outside minutes 1-90:\n{bad.head(10)}"


def test_red_card_minutes_within_1_to_90(red_cards: pd.DataFrame) -> None:
    bad = red_cards[(red_cards.minute < 1) | (red_cards.minute > 90)]
    assert bad.empty, f"{len(bad)} red cards outside minutes 1-90:\n{bad.head(10)}"


def test_each_club_plays_38_matches_per_season_19_home_19_away(matches: pd.DataFrame) -> None:
    problems = []
    for season, group in matches.groupby("season"):
        home_counts = group.home_team.value_counts()
        away_counts = group.away_team.value_counts()
        all_teams = set(group.home_team) | set(group.away_team)
        for team in all_teams:
            h, a = int(home_counts.get(team, 0)), int(away_counts.get(team, 0))
            if h != 19 or a != 19 or h + a != 38:
                problems.append({"season": season, "team": team, "home": h, "away": a, "total": h + a})
    assert not problems, f"{len(problems)} (season, team) combos not at 19 home / 19 away / 38 total: {problems[:10]}"


def test_no_shots_by_a_player_after_his_own_red_card(shots: pd.DataFrame, red_cards: pd.DataFrame) -> None:
    """A sent-off player shouldn't be able to take a shot once he's off the
    pitch. (The match engine picks shot-takers from the full squad with no
    mechanism to exclude a carded player -- see src/squad.py:pick_player and
    src/match_engine.py:_maybe_shot -- so this is expected to surface real
    violations, not just guard against a hypothetical regression.)"""
    violations = []
    for _, card in red_cards.iterrows():
        after = shots[(shots.match_id == card.match_id) & (shots.team == card.team) &
                      (shots.player == card.player) & (shots.minute > card.minute)]
        if len(after):
            violations.append({"match_id": int(card.match_id), "team": card.team, "player": card.player,
                               "sent_off_minute": int(card.minute), "shots_after": len(after)})
    assert not violations, (
        f"{len(violations)} of {len(red_cards)} red-carded players took a shot after their own "
        f"red card (match_id, team, player, sent_off_minute, shots_after): {violations[:10]}")


def test_xg_values_between_0_and_1(shots: pd.DataFrame) -> None:
    bad = shots[(shots.xg < 0) | (shots.xg > 1)]
    assert bad.empty, f"{len(bad)} shots with xG outside [0, 1]:\n{bad.head(10)}"


def test_penalty_xg_mean_close_to_0_76(shots: pd.DataFrame) -> None:
    """Recorded xG is PENALTY_XG_MEAN scaled by that shot's day-form/pressure
    channel multiplier (see src/match_engine.py), which has a slight
    one-sided upward bias on average: pressure spells only ever boost
    (never reduce) a team's output, so the mean multiplier is a bit above
    1.0 (~3-4%), not exactly 1.0. The tolerance here allows for that
    expected shift rather than the raw PENALTY_XG_MEAN value."""
    penalty_xg = shots[shots.is_penalty].xg
    assert len(penalty_xg) > 0, "no penalties found in shots.csv to check"
    mean_xg = penalty_xg.mean()
    assert abs(mean_xg - 0.76) < 0.04, (
        f"mean penalty xG is {mean_xg:.4f} across {len(penalty_xg)} penalties, expected close to 0.76 "
        f"(+ a few % from the day-form/pressure channel multiplier's slight upward bias)")


def test_shot_pre_scores_match_running_score(matches: pd.DataFrame, shots: pd.DataFrame) -> None:
    """Each shot's home_score/away_score should be the score as of just
    BEFORE that shot. Relies on shots.csv preserving the chronological
    generation order per match (ascending minute, home before away within
    a minute) -- verified separately by test_shots_csv_preserves_minute_order."""
    mismatches = []
    for _, m in matches.iterrows():
        match_shots = shots[shots.match_id == m.match_id]
        home_score = away_score = 0
        for _, row in match_shots.iterrows():
            if row.home_score != home_score or row.away_score != away_score:
                mismatches.append({"match_id": int(m.match_id), "minute": int(row.minute),
                                   "team": row.team, "expected": (home_score, away_score),
                                   "recorded": (int(row.home_score), int(row.away_score))})
                break
            if row.is_goal:
                if row.team == m.home_team:
                    home_score += 1
                else:
                    away_score += 1
    assert not mismatches, f"{len(mismatches)} matches with a shot's pre-shot score inconsistent " \
                           f"with the actual running score, e.g.: {mismatches[:10]}"


def test_shots_csv_preserves_minute_order(shots: pd.DataFrame) -> None:
    """Prerequisite for test_shot_pre_scores_match_running_score: within
    each match, rows must appear in non-decreasing minute order (the
    engine appends home's shot before away's shot each minute, so minute
    order alone -- not a full chronological tiebreak -- is what's needed
    and what the running-score check relies on)."""
    bad_matches = []
    for match_id, group in shots.groupby("match_id"):
        if not group.minute.is_monotonic_increasing:
            bad_matches.append(int(match_id))
    assert not bad_matches, f"{len(bad_matches)} matches where shots.csv rows are not in minute order: {bad_matches[:10]}"
