"""Lightweight player squads: ~14 fictional players per club.

Each player has a position, a share of their team's shots (strikers shoot
much more than defenders), a share of their team's red cards, and a
hidden finishing skill (a small multiplier on conversion relative to xG).
No lineups, substitutions, or injuries — one fixed squad per club for the
whole history.
"""

from dataclasses import dataclass

import numpy as np

from src.league import Club

POSITIONS = ["GK"] + ["DF"] * 5 + ["MF"] * 5 + ["FW"] * 3  # 14-player squad shape

SHOT_WEIGHT_BY_POSITION = {"GK": 0.02, "DF": 0.5, "MF": 1.0, "FW": 2.2}
CARD_WEIGHT_BY_POSITION = {"GK": 0.4, "DF": 1.3, "MF": 1.2, "FW": 0.7}

FIRST_NAMES = [
    "Marco", "Luka", "Dario", "Edin", "Tomas", "Nico", "Felix", "Omar",
    "Kwame", "Idris", "Lars", "Mateo", "Rafa", "Sami", "Theo", "Viktor",
    "Erik", "Pavel", "Jonas", "Hugo", "Dion", "Kai", "Noel", "Gabo",
    "Milan", "Enzo", "Aron", "Dax", "Remi", "Joao",
]
LAST_NAMES = [
    "Varga", "Nowak", "Keita", "Santos", "Bauer", "Corbin", "Lindqvist",
    "Osei", "Marek", "Fontaine", "Hartley", "Dubois", "Kovac", "Riva",
    "Alvi", "Brennan", "Soto", "Weiss", "Lund", "Pires", "Castro",
    "Novak", "Bergman", "Horvat", "Delgado", "Mercer", "Vance", "Okafor",
    "Stroud", "Palacios",
]


@dataclass
class Player:
    name: str
    position: str
    shot_share: float       # this player's share of the team's shots
    card_share: float       # this player's share of the team's red cards
    finishing_skill: float  # hidden multiplier on conversion relative to xG


def _random_name(rng: np.random.Generator, used: set[str]) -> str:
    while True:
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if name not in used:
            used.add(name)
            return name


def create_squad(rng: np.random.Generator) -> list[Player]:
    used_names: set[str] = set()
    shot_weights, card_weights, drafts = [], [], []

    for position in POSITIONS:
        shot_weights.append(SHOT_WEIGHT_BY_POSITION[position] * rng.uniform(0.6, 1.4))
        card_weights.append(CARD_WEIGHT_BY_POSITION[position] * rng.uniform(0.7, 1.3))
        drafts.append({
            "name": _random_name(rng, used_names),
            "position": position,
            "finishing_skill": float(np.clip(rng.normal(1.0, 0.12), 0.7, 1.3)),
        })

    shot_total, card_total = sum(shot_weights), sum(card_weights)
    return [
        Player(name=d["name"], position=d["position"], finishing_skill=d["finishing_skill"],
               shot_share=sw / shot_total, card_share=cw / card_total)
        for d, sw, cw in zip(drafts, shot_weights, card_weights)
    ]


def create_squads(clubs: list[Club], seed: int = 1) -> dict[str, list[Player]]:
    """Builds one fixed squad of ~14 players per club, used for the whole history."""
    rng = np.random.default_rng(seed)
    return {club.name: create_squad(rng) for club in clubs}


def pick_player(rng: np.random.Generator, squad: list[Player], by: str) -> Player:
    weights = [getattr(p, by) for p in squad]
    return squad[rng.choice(len(squad), p=weights)]
