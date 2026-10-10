"""Local mock fixtures for testing the dashboard without a live feed."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any


TEAMS = (
    ("MNC", "Manchester City"),
    ("LIV", "Liverpool"),
    ("ARS", "Arsenal"),
    ("CHE", "Chelsea"),
    ("RMA", "Real Madrid"),
    ("BAR", "Barcelona"),
    ("BAY", "Bayern Munich"),
    ("PSG", "Paris Saint-Germain"),
    ("INT", "Inter Milan"),
    ("ATM", "Atletico Madrid"),
    ("DOR", "Borussia Dortmund"),
    ("JUV", "Juventus"),
)


def _decimal_odds(probability: float, margin: float, rng: random.Random) -> float:
    noise = rng.uniform(0.985, 1.015)
    return round(max(1.01, (1 / (probability * margin)) * noise), 2)


def generate_mock_fixtures(count: int = 6, seed: int | None = None) -> list[dict[str, Any]]:
    """Generate virtual-looking fixtures with internally consistent mock odds."""
    if count < 1 or count > 20:
        raise ValueError("Fixture count must be between 1 and 20.")

    rng = random.Random(seed)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fixtures: list[dict[str, Any]] = []
    for index in range(count):
        home, away = rng.sample(TEAMS, 2)
        home_probability = rng.uniform(0.38, 0.61)
        draw_probability = rng.uniform(0.22, 0.31)
        away_probability = max(0.08, 1 - home_probability - draw_probability)
        total = home_probability + draw_probability + away_probability
        probabilities = {
            "home": home_probability / total,
            "draw": draw_probability / total,
            "away": away_probability / total,
        }
        margin = rng.uniform(1.04, 1.12)

        over_probability = rng.uniform(0.63, 0.79)
        totals_margin = rng.uniform(1.04, 1.10)
        fixtures.append(
            {
                "fixture_id": f"mock-{seed or 'local'}-{index + 1}",
                "home_team": home[0],
                "away_team": away[0],
                "home_name": home[1],
                "away_name": away[1],
                "odds": {
                    outcome: _decimal_odds(probability, margin, rng)
                    for outcome, probability in probabilities.items()
                },
                "market_odds": {
                    "over_1_5": _decimal_odds(over_probability, totals_margin, rng),
                    "under_1_5": _decimal_odds(1 - over_probability, totals_margin, rng),
                },
                "source": "simulated demo feed",
                "created_at": created_at,
            }
        )
    return fixtures


def settle_mock_fixture(
    fixture: dict[str, Any],
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Generate a mock final score and outcome for one fixture."""
    rng = rng or random.Random()
    odds = fixture["odds"]
    raw_probabilities = {outcome: 1 / float(odds[outcome]) for outcome in ("home", "draw", "away")}
    normalizer = sum(raw_probabilities.values())
    outcomes = ("home", "draw", "away")
    actual = rng.choices(
        outcomes,
        weights=[raw_probabilities[outcome] / normalizer for outcome in outcomes],
        k=1,
    )[0]

    if actual == "draw":
        home_score = away_score = rng.randint(0, 3)
    elif actual == "home":
        away_score = rng.randint(0, 2)
        home_score = rng.randint(away_score + 1, away_score + 3)
    else:
        home_score = rng.randint(0, 2)
        away_score = rng.randint(home_score + 1, home_score + 3)

    return {
        "fixture_id": fixture["fixture_id"],
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "outcome": actual,
        "score": f"{home_score}-{away_score}",
        "settled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }