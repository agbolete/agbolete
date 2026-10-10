"""Transparent, odds-derived estimates for Instant Virtual football markets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class InstantVirtualPredictor:
    """Analyze decimal odds without presenting heuristics as guaranteed outcomes.

    The built-in baseline is a generic reference distribution, not a proprietary
    or empirically verified SportyBet model. Supply a calibrated baseline only
    when it has been derived from a sufficiently large, lawful result dataset.
    """

    DEFAULT_BASELINE = {"home": 0.45, "draw": 0.27, "away": 0.28}

    def __init__(
        self,
        *,
        baseline: Mapping[str, float] | None = None,
        max_overround: float = 1.20,
        min_overround: float = 0.98,
        baseline_gap_threshold: float = 0.16,
        tight_draw_gap: float = 0.025,
        slump_losses: int = 3,
        slump_penalty: float = 0.04,
        strong_signal_threshold: float = 0.62,
        stable_signal_threshold: float = 0.50,
    ) -> None:
        self.baseline = dict(baseline or self.DEFAULT_BASELINE)
        if set(self.baseline) != {"home", "draw", "away"}:
            raise ValueError("Baseline must define home, draw, and away probabilities.")
        baseline_total = sum(float(value) for value in self.baseline.values())
        if any(value <= 0 or not math.isfinite(value) for value in self.baseline.values()):
            raise ValueError("Baseline probabilities must be finite and positive.")
        self.baseline = {key: float(value) / baseline_total for key, value in self.baseline.items()}
        self.max_overround = max_overround
        self.min_overround = min_overround
        self.baseline_gap_threshold = baseline_gap_threshold
        self.tight_draw_gap = tight_draw_gap
        self.slump_losses = slump_losses
        self.slump_penalty = slump_penalty
        self.strong_signal_threshold = strong_signal_threshold
        self.stable_signal_threshold = stable_signal_threshold

    @staticmethod
    def _probabilities_from_odds(odds: Mapping[str, Any]) -> tuple[dict[str, float], float]:
        outcomes = ("home", "draw", "away")
        try:
            parsed = {outcome: float(odds[outcome]) for outcome in outcomes}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("A complete decimal-odds set for home, draw, and away is required.") from exc
        if any(not math.isfinite(value) or value <= 1.0 or value > 100.0 for value in parsed.values()):
            raise ValueError("Odds must be finite decimal values greater than 1 and at most 100.")

        raw = {outcome: 1.0 / value for outcome, value in parsed.items()}
        overround = sum(raw.values())
        return ({outcome: probability / overround for outcome, probability in raw.items()}, overround)

    @staticmethod
    def _form_results(value: str | Sequence[str] | None) -> list[str]:
        if value is None:
            return []
        characters = list(value) if not isinstance(value, str) else list(value.upper())
        return [character.upper() for character in characters if str(character).upper() in {"W", "D", "L"}][-5:]

    @staticmethod
    def _adjust_for_form(
        probabilities: Mapping[str, float],
        home_team: str,
        away_team: str,
        team_form: Mapping[str, str | Sequence[str]] | None,
        slump_losses: int,
        slump_penalty: float,
    ) -> tuple[dict[str, float], list[str], dict[str, list[str]]]:
        adjusted = dict(probabilities)
        forms = {str(code).upper(): value for code, value in (team_form or {}).items()}
        recent = {
            home_team: InstantVirtualPredictor._form_results(forms.get(home_team)),
            away_team: InstantVirtualPredictor._form_results(forms.get(away_team)),
        }
        slumping = [team for team, results in recent.items() if results.count("L") >= slump_losses]

        for team in slumping:
            side = "home" if team == home_team else "away"
            reduction = min(slump_penalty, adjusted[side] / 2)
            adjusted[side] -= reduction
            other_sides = [name for name in adjusted if name != side]
            other_total = sum(adjusted[name] for name in other_sides)
            for other_side in other_sides:
                adjusted[other_side] += reduction * adjusted[other_side] / other_total

        return adjusted, slumping, recent

    def analyze_fixture(
        self,
        fixture: Mapping[str, Any],
        team_form: Mapping[str, str | Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        """Return 1X2, double-chance, optional totals, review flags, and a signal."""
        home_team = str(fixture.get("home_team", fixture.get("home", "HOME"))).upper()
        away_team = str(fixture.get("away_team", fixture.get("away", "AWAY"))).upper()
        odds = fixture.get("odds")
        if not isinstance(odds, Mapping):
            raise ValueError("Fixture odds must be provided as a mapping.")

        probabilities, overround = self._probabilities_from_odds(odds)
        flags: list[str] = []
        if overround > self.max_overround:
            flags.append("High implied bookmaker margin; verify this fixture.")
        elif overround < self.min_overround:
            flags.append("Unusually low implied bookmaker margin; verify this fixture.")

        baseline_gaps = {
            outcome: probabilities[outcome] - self.baseline[outcome]
            for outcome in probabilities
        }
        large_gaps = [
            outcome.upper()
            for outcome, gap in baseline_gaps.items()
            if abs(gap) >= self.baseline_gap_threshold
        ]
        if large_gaps:
            flags.append("Large gap versus generic reference: " + ", ".join(large_gaps) + ".")

        if (
            abs(probabilities["draw"] - probabilities["home"]) < self.tight_draw_gap
            or abs(probabilities["draw"] - probabilities["away"]) < self.tight_draw_gap
        ):
            flags.append("Tightly priced draw; treat the 1X2 spread cautiously.")

        adjusted, slump_teams, recent_form = self._adjust_for_form(
            probabilities,
            home_team,
            away_team,
            team_form,
            self.slump_losses,
            self.slump_penalty,
        )

        markets: dict[str, dict[str, float | str]] = {
            "home_win": {"probability": probabilities["home"], "basis": "normalized 1X2 odds"},
            "draw": {"probability": probabilities["draw"], "basis": "normalized 1X2 odds"},
            "away_win": {"probability": probabilities["away"], "basis": "normalized 1X2 odds"},
            "home_or_draw": {
                "probability": probabilities["home"] + probabilities["draw"],
                "basis": "normalized 1X2 odds",
            },
            "away_or_draw": {
                "probability": probabilities["away"] + probabilities["draw"],
                "basis": "normalized 1X2 odds",
            },
            "home_or_away": {
                "probability": probabilities["home"] + probabilities["away"],
                "basis": "normalized 1X2 odds",
            },
        }

        market_odds = fixture.get("market_odds", {})
        if isinstance(market_odds, Mapping):
            over = self._optional_decimal(market_odds.get("over_1_5"))
            under = self._optional_decimal(market_odds.get("under_1_5"))
            if over is not None and under is not None:
                total_raw = {"over": 1.0 / over, "under": 1.0 / under}
                total_margin = sum(total_raw.values())
                markets["over_1_5_goals"] = {
                    "probability": total_raw["over"] / total_margin,
                    "basis": "normalized over/under odds",
                }
                markets["under_1_5_goals"] = {
                    "probability": total_raw["under"] / total_margin,
                    "basis": "normalized over/under odds",
                }

        best_outcome = max(adjusted, key=adjusted.get)
        best_probability = adjusted[best_outcome]
        if flags:
            recommendation = "❌ SKIP: VERIFY MARKET"
        elif best_probability >= self.strong_signal_threshold:
            recommendation = f"🔥 STRONG BUY: {best_outcome.upper()} WIN"
        elif best_probability >= self.stable_signal_threshold:
            recommendation = "✅ STABLE: " + best_outcome.upper()
        else:
            recommendation = "❌ SKIP: NO CLEAR EDGE"

        return {
            "fixture": f"{home_team} vs {away_team}",
            "home_team": home_team,
            "away_team": away_team,
            "decimal_odds": {key: float(odds[key]) for key in ("home", "draw", "away")},
            "overround": overround,
            "odds_probabilities": probabilities,
            "form_adjusted_probabilities": adjusted,
            "baseline_probabilities": dict(self.baseline),
            "baseline_gaps": baseline_gaps,
            "markets": markets,
            "recent_form": recent_form,
            "slump_teams": slump_teams,
            "recommendation": recommendation,
            "recommendation_probability": best_probability,
            "review_required": bool(flags),
            "review_flags": flags,
            "disclaimer": (
                "Odds-derived estimates and generic heuristics only; not a verified SportyBet model, "
                "guarantee, or betting advice."
            ),
        }

    @staticmethod
    def _optional_decimal(value: Any) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) and 1.0 < parsed <= 100.0 else None

    def analyze_matches(
        self,
        fixtures: Sequence[Mapping[str, Any]],
        team_form: Mapping[str, str | Sequence[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze valid fixtures, returning parse errors alongside their fixture."""
        reports = []
        for fixture in fixtures:
            try:
                reports.append(self.analyze_fixture(fixture, team_form))
            except (TypeError, ValueError) as exc:
                reports.append({"fixture": str(fixture.get("raw_line", "Unknown fixture")), "error": str(exc)})
        return reports