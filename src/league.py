"""Defines the fictional league: 20 clubs, each with hidden 'true' traits.

These traits are only used by the match engine to generate data. Later,
we'll try to recover them purely from the generated season stats, without
ever reading them directly.

Traits also drift slowly from season to season (squads change), with an
occasional bigger jump representing something like a new coach.
"""

from dataclasses import dataclass, replace

import numpy as np

CLUB_NAMES = [
    "Northgate United",
    "Port Albion",
    "Red Valley FC",
    "Stonebridge Town",
    "Ashford Rovers",
    "West Harbor City",
    "Kingswell Athletic",
    "Millbrook Town",
    "Eastcliff FC",
    "Thornfield United",
    "Bridgeport Rovers",
    "Oakhaven City",
    "Silverlake FC",
    "Greymoor Athletic",
    "Fenwick United",
    "Harrow Vale",
    "Dunmore Town",
    "Castlebay FC",
    "Riversedge United",
    "Blackthorn Rovers",
]

# Valid range for each trait, enforced after both initial creation and drift.
TRAIT_BOUNDS = {
    "attack": (0.35, 1.65),
    "defense": (0.35, 1.65),
    "comeback_factor": (1.0, 1.5),
    "protect_lead_factor": (1.0, 1.5),
    "ten_men_resilience": (0.5, 0.98),
    "fatigue_factor": (0.6, 1.0),
}


@dataclass
class Club:
    name: str
    attack: float              # scoring strength; league-average is 1.0
    defense: float             # defensive strength; league-average is 1.0
    comeback_factor: float     # attack boost while trailing (>=1.0)
    protect_lead_factor: float  # defense boost while leading (>=1.0)
    ten_men_resilience: float  # attack & defense multiplier with 10 men (<=1.0)
    fatigue_factor: float      # defense multiplier after minute 75 (<=1.0)


def create_league(seed: int = 42) -> list[Club]:
    """Builds the 20 clubs with randomized hidden traits."""
    rng = np.random.default_rng(seed)
    clubs = []
    for name in CLUB_NAMES:
        clubs.append(
            Club(
                name=name,
                attack=rng.uniform(0.6, 1.5),
                defense=rng.uniform(0.6, 1.5),
                comeback_factor=rng.uniform(1.0, 1.3),
                protect_lead_factor=rng.uniform(1.0, 1.3),
                ten_men_resilience=rng.uniform(0.6, 0.95),
                fatigue_factor=rng.uniform(0.75, 1.0),
            )
        )
    return clubs


def drift_traits(club: Club, rng: np.random.Generator, jump_prob: float = 0.08,
                  drift_sigma: float = 0.03, jump_sigma: float = 0.15) -> Club:
    """Ages a club by one season. Each trait takes a small random-walk step;
    occasionally (jump_prob) the club instead takes a much bigger step,
    standing in for something like a new coach or a squad overhaul."""
    sigma = jump_sigma if rng.random() < jump_prob else drift_sigma
    updates = {
        trait: float(np.clip(getattr(club, trait) + rng.normal(0, sigma), lo, hi))
        for trait, (lo, hi) in TRAIT_BOUNDS.items()
    }
    return replace(club, **updates)
