"""Shared fixtures for the test suite. Session-scoped wherever the
underlying data/model doesn't change between tests, so the (sizeable)
season data and model files are only read from disk once per test run.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.win_probability import clubs_from_traits

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def matches() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "matches.csv")


@pytest.fixture(scope="session")
def shots() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "shots.csv")


@pytest.fixture(scope="session")
def goals_csv() -> pd.DataFrame:
    """The actual saved data/goals.csv -- distinct from shots[shots.is_goal],
    which is a derived view used to cross-check the saved file (see
    test_data_integrity.test_goals_csv_matches_shots_is_goal_filter)."""
    return pd.read_csv(DATA_DIR / "goals.csv")


@pytest.fixture(scope="session")
def red_cards() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "red_cards.csv")


@pytest.fixture(scope="session")
def players() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "players.csv")


@pytest.fixture(scope="session")
def estimated_traits() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "estimated_traits.csv").set_index("team")


@pytest.fixture(scope="session")
def latest_season(matches: pd.DataFrame) -> int:
    return int(matches.season.max())


@pytest.fixture(scope="session")
def season5_matches(matches: pd.DataFrame, latest_season: int) -> pd.DataFrame:
    return matches[matches.season == latest_season].reset_index(drop=True)


@pytest.fixture(scope="session")
def season5_shots(shots: pd.DataFrame, season5_matches: pd.DataFrame) -> pd.DataFrame:
    ids = set(season5_matches.match_id)
    return shots[shots.match_id.isin(ids)].reset_index(drop=True)


@pytest.fixture(scope="session")
def season5_red_cards(red_cards: pd.DataFrame, season5_matches: pd.DataFrame) -> pd.DataFrame:
    ids = set(season5_matches.match_id)
    return red_cards[red_cards.match_id.isin(ids)].reset_index(drop=True)


@pytest.fixture(scope="session")
def point_clubs(estimated_traits: pd.DataFrame) -> dict:
    """Point-estimate Club objects, one per team -- the fast, single-draw
    model used for exhaustive checks across many matches/events."""
    return clubs_from_traits(estimated_traits)


@pytest.fixture(scope="session")
def team_draws(data_dir: Path) -> list[dict]:
    """The precomputed parameter-uncertainty trait draws (see
    scripts/precompute_featured_matches.py) -- used sparingly, for the
    smaller-scale checks that need the full production model rather than
    the point estimate."""
    df = pd.read_csv(data_dir / "model_team_draws.csv")
    return [clubs_from_traits(group.set_index("team")) for _, group in df.groupby("draw")]


@pytest.fixture(scope="session")
def featured_curves(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "featured_curves.csv")


@pytest.fixture(scope="session")
def featured_turning_points(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "featured_turning_points.csv")
    df["secondary_events"] = df.secondary_events.apply(json.loads)
    return df


@pytest.fixture(scope="session")
def featured_meta(data_dir: Path) -> dict:
    return json.loads((data_dir / "featured_matches.json").read_text())
