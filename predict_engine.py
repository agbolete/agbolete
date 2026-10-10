"""League-aware 1X2 probability estimates for Virtual Football fixtures.

Team ratings and adjustments here are transparent heuristics, not a trained or
verified SportyBet probability model. Percentages are estimates, not guarantees.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


ELITE_TIER = {
    "MNC": 86, "LIV": 86, "ARS": 84, "RMA": 87, "BAR": 87,
    "INT": 84, "JUV": 84,
}
UPPER_TIER = {
    "CHE": 80, "TOT": 80, "NEW": 79, "ATM": 83, "SEV": 78,
    "VIL": 78, "SOC": 77, "MIL": 81, "NAP": 82, "ROM": 79,
    "DOR": 81, "LEI": 78, "LEV": 82,
}
MID_TIER = {
    "AST": 74, "BHA": 73, "WHU": 72, "WOL": 71, "FUL": 70,
    "CRY": 70, "BOU": 69, "EVE": 70, "BIL": 74, "VAL": 70,
    "LAZ": 74, "ATA": 74, "FIO": 72, "FRA": 70, "WOB": 71,
}
LOWER_TIER = {"NFO": 63, "BRE": 64, "LUT": 60, "BUR": 58, "SHU": 57}

# MUN intentionally has separate values because the same code may mean either club.
MUN_STRENGTH_BY_LEAGUE = {"england": 79, "germany": 86}
TEAM_STRENGTH_INDEX = {
    **ELITE_TIER,
    **UPPER_TIER,
    **MID_TIER,
    **LOWER_TIER,
    "MUN": MUN_STRENGTH_BY_LEAGUE,
}
TEAM_LEAGUES = {
    **{code: "england" for code in ("MNC", "LIV", "ARS", "CHE", "TOT", "NEW", "AST", "BHA", "WHU", "WOL", "FUL", "CRY", "BOU", "EVE", "NFO", "BRE", "LUT", "BUR", "SHU")},
    **{code: "spain" for code in ("RMA", "BAR", "ATM", "SEV", "VIL", "SOC", "BIL", "VAL")},
    **{code: "italy" for code in ("MIL", "INT", "JUV", "NAP", "ROM", "LAZ", "ATA", "FIO")},
    **{code: "germany" for code in ("DOR", "LEI", "LEV", "FRA", "WOB")},
}


class InstantVirtualPredictor:
    """Compute distinct 1X2 estimates, generic strength priors, and risk flags."""

    def __init__(
        self,
        *,
        strength_weight: float = 0.25,
        home_advantage_weight: float = 0.03,
        compact_draw_odds: tuple[float, float] = (3.10, 3.50),
        close_rating_gap: float = 6.0,
        compact_draw_boost: float = 0.035,
        strong_edge_threshold: float = 0.05,
        stable_edge_threshold: float = 0.02,
        max_overround: float = 1.20,
        min_overround: float = 0.98,
    ) -> None:
        if not 0 <= strength_weight <= 1:
            raise ValueError("strength_weight must be between 0 and 1.")
        self.strength_weight = strength_weight
        self.home_advantage_weight = home_advantage_weight
        self.compact_draw_odds = compact_draw_odds
        self.close_rating_gap = close_rating_gap
        self.compact_draw_boost = compact_draw_boost
        self.strong_edge_threshold = strong_edge_threshold
        self.stable_edge_threshold = stable_edge_threshold
        self.max_overround = max_overround
        self.min_overround = min_overround

    @staticmethod
    def _normalize_league(value: Any) -> str | None:
        text = str(value or "").strip().lower()
        if not text:
            return None
        aliases = {
            "england": ("england", "english", "premier", "epl"),
            "spain": ("spain", "spanish", "la liga", "laliga"),
            "italy": ("italy", "italian", "serie a"),
            "germany": ("germany", "german", "bundesliga"),
        }
        for league, candidates in aliases.items():
            if any(candidate in text for candidate in candidates):
                return league
        return None

    @classmethod
    def _team_strength(
        cls,
        code: str,
        league_hint: Any = None,
        opponent_code: str | None = None,
        team_name: str | None = None,
    ) -> tuple[float, bool, str | None]:
        normalized_code = str(code).upper().strip()
        explicit_league = cls._normalize_league(league_hint)
        normalized_name = str(team_name or "").lower()

        if normalized_code == "MUN":
            if "bayern" in normalized_name:
                return float(MUN_STRENGTH_BY_LEAGUE["germany"]), False, "germany"
            if "manchester" in normalized_name or "man united" in normalized_name:
                return float(MUN_STRENGTH_BY_LEAGUE["england"]), False, "england"
            resolved_league = explicit_league
            if resolved_league not in MUN_STRENGTH_BY_LEAGUE and opponent_code:
                resolved_league = TEAM_LEAGUES.get(str(opponent_code).upper())
            if resolved_league in MUN_STRENGTH_BY_LEAGUE:
                return float(MUN_STRENGTH_BY_LEAGUE[resolved_league]), False, resolved_league
            return sum(MUN_STRENGTH_BY_LEAGUE.values()) / 2, True, None

        strength = TEAM_STRENGTH_INDEX.get(normalized_code)
        if isinstance(strength, (int, float)):
            return float(strength), False, TEAM_LEAGUES.get(normalized_code)
        return 70.0, True, None

    @staticmethod
    def _odds_mapping(match_config: Mapping[str, Any]) -> dict[str, float]:
        source = match_config.get("odds", match_config)
        if not isinstance(source, Mapping):
            raise ValueError("Odds must be provided as a mapping.")
        aliases = {
            "home": ("home", "home_odds", "h", "1"),
            "draw": ("draw", "draw_odds", "d", "x"),
            "away": ("away", "away_odds", "a", "2"),
        }
        parsed: dict[str, float] = {}
        normalized = {str(key).lower(): value for key, value in source.items()}
        for outcome, names in aliases.items():
            raw = next((normalized[name] for name in names if name in normalized), None)
            if raw is None:
                raise ValueError(f"Missing {outcome} decimal odds.")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {outcome} decimal odds.") from exc
            if not math.isfinite(value) or not 1.0 < value <= 100:
                raise ValueError(f"{outcome.title()} odds must be finite and between 1 and 100.")
            parsed[outcome] = value
        return parsed

    @staticmethod
    def _market_probabilities(odds: Mapping[str, float]) -> tuple[dict[str, float], float]:
        raw = {outcome: 1.0 / odds[outcome] for outcome in ("home", "draw", "away")}
        overround = sum(raw.values())
        return ({outcome: probability / overround for outcome, probability in raw.items()}, overround)

    @staticmethod
    def _strength_prior(rating_gap: float) -> dict[str, float]:
        draw = 0.27 * math.exp(-abs(rating_gap) / 45.0)
        decisive_share = 1.0 - draw
        home_share = 1.0 / (1.0 + math.exp(-rating_gap / 14.0))
        return {
            "home": decisive_share * home_share,
            "draw": draw,
            "away": decisive_share * (1.0 - home_share),
        }

    @staticmethod
    def _normalize(probabilities: Mapping[str, float]) -> dict[str, float]:
        total = sum(probabilities.values())
        if total <= 0 or not math.isfinite(total):
            raise ValueError("Probability calculation produced an invalid total.")
        return {outcome: value / total for outcome, value in probabilities.items()}

    @staticmethod
    def _apply_probability_transfer(
        probabilities: Mapping[str, float],
        target: str,
        amount: float,
    ) -> dict[str, float]:
        adjusted = dict(probabilities)
        amount = min(max(0.0, amount), 1.0 - adjusted[target])
        donors = [outcome for outcome in adjusted if outcome != target]
        donor_total = sum(adjusted[outcome] for outcome in donors)
        if donor_total <= 0:
            return adjusted
        adjusted[target] += amount
        for outcome in donors:
            adjusted[outcome] -= amount * adjusted[outcome] / donor_total
        return adjusted

    @staticmethod
    def _form_results(value: str | Sequence[str] | None) -> list[str]:
        if value is None:
            return []
        values = list(value) if not isinstance(value, str) else list(value.upper())
        return [str(result).upper() for result in values if str(result).upper() in {"W", "D", "L"}][-5:]

    def calculate_probabilities(
        self,
        match_config: Mapping[str, Any],
        *,
        home_advantage: bool | None = None,
        team_form: Mapping[str, str | Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        """Return requested percentage fields plus detail and legacy app fields.

        ``home_advantage`` may be provided explicitly or set on the match as
        ``home_advantage`` / ``simulation_cycle``. Without cycle metadata, no
        home adjustment is assumed.
        """
        home_code = str(match_config.get("home_team", match_config.get("home_code", match_config.get("home", "HOME")))).upper()
        away_code = str(match_config.get("away_team", match_config.get("away_code", match_config.get("away", "AWAY")))).upper()
        home_name = str(match_config.get("home_name", ""))
        away_name = str(match_config.get("away_name", ""))
        league_hint = match_config.get("league", match_config.get("competition"))

        odds = self._odds_mapping(match_config)
        market_probabilities, overround = self._market_probabilities(odds)
        home_strength, home_ambiguous, home_league = self._team_strength(
            home_code, match_config.get("home_league", league_hint), away_code, home_name
        )
        away_strength, away_ambiguous, away_league = self._team_strength(
            away_code, match_config.get("away_league", league_hint), home_code, away_name
        )
        strength_gap = home_strength - away_strength
        prior = self._strength_prior(strength_gap)

        estimates = {
            outcome: (1.0 - self.strength_weight) * market_probabilities[outcome]
            + self.strength_weight * prior[outcome]
            for outcome in ("home", "draw", "away")
        }
        estimates = self._normalize(estimates)

        cycle_value = match_config.get("simulation_cycle")
        if home_advantage is None:
            home_advantage = bool(match_config.get("home_advantage", cycle_value))
        if home_advantage:
            estimates = self._apply_probability_transfer(
                estimates, "home", self.home_advantage_weight
            )

        draw_in_compact_range = self.compact_draw_odds[0] <= odds["draw"] <= self.compact_draw_odds[1]
        closely_rated = abs(strength_gap) <= self.close_rating_gap
        draw_boost_applied = draw_in_compact_range and closely_rated
        if draw_boost_applied:
            estimates = self._apply_probability_transfer(
                estimates, "draw", self.compact_draw_boost
            )

        form_map = {str(code).upper(): value for code, value in (team_form or {}).items()}
        recent_form = {
            home_code: self._form_results(form_map.get(home_code)),
            away_code: self._form_results(form_map.get(away_code)),
        }
        slump_teams = [
            code
            for code, results in recent_form.items()
            if results.count("L") >= 3
        ]
        for team in slump_teams:
            outcome = "home" if team == home_code else "away"
            penalty = min(0.04, estimates[outcome] / 2)
            estimates[outcome] -= penalty
            other_outcomes = [item for item in estimates if item != outcome]
            other_total = sum(estimates[item] for item in other_outcomes)
            for item in other_outcomes:
                estimates[item] += penalty * estimates[item] / other_total
        estimates = self._normalize(estimates)

        edges = {
            outcome: estimates[outcome] - market_probabilities[outcome]
            for outcome in ("home", "away", "draw")
        }
        best_edge_outcome = max(edges, key=edges.get)
        best_edge = edges[best_edge_outcome]

        risk_flags: list[str] = []
        if overround > self.max_overround:
            risk_flags.append("High bookmaker margin; verify extracted odds.")
        elif overround < self.min_overround:
            risk_flags.append("Unusually low implied margin; verify extracted odds.")
        if home_ambiguous or away_ambiguous:
            risk_flags.append("MUN team identity is ambiguous without a league heading or name.")
        unknown_codes = [
            code for code, rating, ambiguous in (
                (home_code, home_strength, home_ambiguous),
                (away_code, away_strength, away_ambiguous),
            )
            if rating == 70.0 and code not in TEAM_STRENGTH_INDEX and not ambiguous
        ]
        if unknown_codes:
            risk_flags.append("No configured strength tier for " + ", ".join(unknown_codes) + ".")
        ocr_confidence = match_config.get("ocr_confidence")
        if ocr_confidence is not None:
            try:
                if float(ocr_confidence) < 0.25:
                    risk_flags.append("Low OCR confidence; verify the parsed team codes and odds.")
            except (TypeError, ValueError):
                risk_flags.append("Invalid OCR confidence metadata.")
        if max(abs(value) for value in edges.values()) >= 0.20:
            risk_flags.append("Large divergence from bookmaker-implied probabilities.")

        highest_probability_outcome = max(estimates, key=estimates.get)
        signal = {
            "home": "HOME WIN",
            "away": "AWAY WIN",
            "draw": "DRAW",
        }[highest_probability_outcome]

        market_odds = match_config.get("market_odds", {})
        markets: dict[str, dict[str, float | str]] = {
            "home_win": {"probability": estimates["home"], "basis": "odds and strength heuristic"},
            "draw": {"probability": estimates["draw"], "basis": "odds and strength heuristic"},
            "away_win": {"probability": estimates["away"], "basis": "odds and strength heuristic"},
            "home_or_draw": {"probability": estimates["home"] + estimates["draw"], "basis": "derived 1X2"},
            "away_or_draw": {"probability": estimates["away"] + estimates["draw"], "basis": "derived 1X2"},
            "home_or_away": {"probability": estimates["home"] + estimates["away"], "basis": "derived 1X2"},
        }
        if isinstance(market_odds, Mapping):
            over = self._optional_decimal(market_odds.get("over_1_5"))
            under = self._optional_decimal(market_odds.get("under_1_5"))
            if over is not None and under is not None:
                total_raw = {"over": 1 / over, "under": 1 / under}
                total_sum = sum(total_raw.values())
                markets["over_1_5_goals"] = {
                    "probability": total_raw["over"] / total_sum,
                    "basis": "normalized Over/Under odds",
                }
                markets["under_1_5_goals"] = {
                    "probability": total_raw["under"] / total_sum,
                    "basis": "normalized Over/Under odds",
                }

        fixture_name = f"{home_name or home_code} vs {away_name or away_code}"
        percentage_values = {
            "Home_Win_PC": round(estimates["home"] * 100, 2),
            "Away_Win_PC": round(estimates["away"] * 100, 2),
            "Draw_PC": round(estimates["draw"] * 100, 2),
        }
        return {
            **percentage_values,
            "Staking_Signal": signal,
            "fixture": fixture_name,
            "home_team": home_code,
            "away_team": away_code,
            "odds_probabilities": estimates,
            "market_implied_probabilities": market_probabilities,
            "strength_prior_probabilities": prior,
            "strength_gap": strength_gap,
            "team_strength": {"home": home_strength, "away": away_strength},
            "resolved_leagues": {"home": home_league, "away": away_league},
            "edges": {key: round(value * 100, 2) for key, value in edges.items()},
            "best_edge_market": best_edge_outcome,
            "recommendation": signal,
            "recommendation_probability": estimates[best_edge_outcome],
            "overround": overround,
            "markets": markets,
            "slump_teams": slump_teams,
            "recent_form": recent_form,
            "home_advantage_applied": bool(home_advantage),
            "draw_boost_applied": draw_boost_applied,
            "review_required": bool(risk_flags),
            "review_flags": risk_flags,
            "disclaimer": (
                "Heuristic odds-derived estimates only; ratings are not an official or empirically "
                "verified SportyBet model. This is not a guarantee or betting advice."
            ),
        }

    def analyze_fixture(
        self,
        fixture: Mapping[str, Any],
        team_form: Mapping[str, str | Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        """Backward-compatible wrapper used by the existing Streamlit app."""
        return self.calculate_probabilities(fixture, team_form=team_form)

    def analyze_matches(
        self,
        fixtures: Sequence[Mapping[str, Any]],
        team_form: Mapping[str, str | Sequence[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze a list of screenshot or mock-feed fixtures independently."""
        results = []
        for fixture in fixtures:
            try:
                results.append(self.calculate_probabilities(fixture, team_form=team_form))
            except (TypeError, ValueError) as exc:
                results.append({"fixture": str(fixture.get("raw_line", "Unknown fixture")), "error": str(exc)})
        return results

    @staticmethod
    def _optional_decimal(value: Any) -> float | None:
        if value is None:
            return None
        try:
            decimal = float(value)
        except (TypeError, ValueError):
            return None
        return decimal if math.isfinite(decimal) and 1.0 < decimal <= 100.0 else None


__all__ = [
    "ELITE_TIER",
    "UPPER_TIER",
    "MID_TIER",
    "LOWER_TIER",
    "MUN_STRENGTH_BY_LEAGUE",
    "TEAM_STRENGTH_INDEX",
    "InstantVirtualPredictor",
]