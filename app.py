"""SportyBet Instant Virtual screenshot analysis dashboard."""

from __future__ import annotations

import base64
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from supabase import create_client

from image_processor import ScreenshotProcessingError, extract_matches
from mock_feed import generate_mock_fixtures, settle_mock_fixture
from predict_engine import InstantVirtualPredictor


st.set_page_config(
    page_title="Instant Virtual Analyzer",
    page_icon=":material/sports_soccer:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
    st.error(
        "Configuration Error: Missing keys. Ensure your credentials live inside "
        ".streamlit/secrets.toml at the root layer."
    )
    st.stop()

SUPABASE_URL = str(st.secrets["SUPABASE_URL"]).strip()
SUPABASE_KEY = str(st.secrets["SUPABASE_KEY"]).strip()


def validate_supabase_settings(url: str, key: str) -> str | None:
    if not url.startswith("https://") or not key:
        return "Supabase URL or public anon key is empty or malformed."
    if "PASTE_YOUR_" in url or "PASTE_YOUR_" in key:
        return "Replace the Supabase placeholder values in .streamlit/secrets.toml."
    try:
        payload = key.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError, json.JSONDecodeError):
        return None

    if claims.get("role") == "service_role":
        return "Do not use a service_role key in this app. Use the public anon/publishable key."
    host = urlparse(url).hostname or ""
    url_ref = host.split(".")[0]
    token_ref = claims.get("ref")
    if token_ref and token_ref != url_ref:
        return "Supabase URL and anon key belong to different projects. Use a matching URL/key pair."
    return None


configuration_error = validate_supabase_settings(SUPABASE_URL, SUPABASE_KEY)
if configuration_error:
    st.error("Configuration Error: " + configuration_error)
    st.stop()

st.markdown(
    """
    <style>
    [data-testid="stMain"] { max-width: 780px; padding-top: 1rem; }
    [data-testid="stMetric"] {
        background: #171c19;
        border: 1px solid #2b3430;
        border-radius: 8px;
        padding: .85rem;
        min-height: 106px;
    }
    [data-testid="stMetricLabel"] { font-size: .8rem; color: #aab2ac; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    .app-kicker { color: #c8f169; font-size: .76rem; font-weight: 700; }
    @media (max-width: 520px) {
        [data-testid="stMain"] { padding: .8rem .65rem 1.5rem; }
        [data-testid="stMetric"] { padding: .7rem; min-height: 98px; }
        [data-testid="stMetricValue"] { font-size: 1.15rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def save_auth_session(session: Any) -> None:
    if session is not None:
        st.session_state.auth_tokens = {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
        }


def current_user(client: Any) -> Any | None:
    tokens = st.session_state.get("auth_tokens")
    if not tokens:
        return None
    try:
        client.auth.set_session(tokens["access_token"], tokens["refresh_token"])
        response = client.auth.get_user()
        save_auth_session(client.auth.get_session())
        return response.user
    except Exception:
        st.session_state.pop("auth_tokens", None)
        return None


def render_auth(client: Any) -> None:
    st.markdown('<div class="app-kicker">SPORTYBET INSTANT VIRTUAL</div>', unsafe_allow_html=True)
    st.title("Your match analyzer")
    st.caption("Create an account or sign in to analyze virtual match screenshots.")
    mode = st.segmented_control(
        "Account action",
        ["Sign in", "Create account"],
        default="Sign in",
        key="auth_mode",
    )

    if mode == "Create account":
        with st.form("registration_form"):
            email = st.text_input("Email address", autocomplete="email")
            password = st.text_input(
                "Password",
                type="password",
                autocomplete="new-password",
                help="Use at least 12 characters and avoid reusing a password.",
            )
            phone = st.text_input(
                "SportyBet account phone number",
                autocomplete="tel",
                placeholder="Include your country code, e.g. +234...",
            )
            st.caption(
                "Supabase Auth receives the password over HTTPS and stores a password hash. "
                "The phone number is profile metadata; this app does not access your SportyBet account."
            )
            submitted = st.form_submit_button(
                "Create account", type="primary", icon=":material/person_add:"
            )

        if submitted:
            clean_phone = re.sub(r"[\s()-]", "", phone)
            if not email.strip() or "@" not in email:
                st.error("Enter a valid email address.")
            elif len(password) < 12:
                st.error("Use a password with at least 12 characters.")
            elif not re.fullmatch(r"\+?[1-9]\d{6,14}", clean_phone):
                st.error("Enter a valid phone number with country code.")
            else:
                try:
                    response = client.auth.sign_up(
                        {
                            "email": email.strip(),
                            "password": password,
                            "options": {"data": {"sportybet_phone": clean_phone}},
                        }
                    )
                    if response.session:
                        save_auth_session(response.session)
                        st.rerun()
                    st.success("Registration submitted. Confirm your email, then sign in.")
                except Exception:
                    st.error("Registration failed. Check your email and Supabase Auth settings.")
    else:
        with st.form("login_form"):
            email = st.text_input("Email address", autocomplete="email")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", type="primary", icon=":material/login:")

        if submitted:
            try:
                response = client.auth.sign_in_with_password(
                    {"email": email.strip(), "password": password}
                )
                save_auth_session(response.session)
                st.rerun()
            except Exception:
                st.error("Sign-in failed. Check your details or confirm your email address.")


def render_prediction(report: dict[str, Any]) -> None:
    if "error" in report:
        st.warning(f"{report['fixture']}: {report['error']}")
        return

    with st.container(border=True):
        st.subheader(report["fixture"])
        st.metric("Signal", report["recommendation"])
        st.caption(
            f"Top outcome: {report['recommendation_probability']:.1%} · "
            f"Implied margin: {report['overround']:.1%}"
        )
        first, second = st.columns(2)
        for column, outcome, label in (
            (first, "home", f"{report['home_team']} win"),
            (second, "draw", "Draw"),
            (first, "away", f"{report['away_team']} win"),
        ):
            column.metric(label, f"{report['odds_probabilities'][outcome]:.1%}")

        markets = report["markets"]
        st.caption(
            f"Double chance: 1X {markets['home_or_draw']['probability']:.1%} · "
            f"X2 {markets['away_or_draw']['probability']:.1%} · "
            f"12 {markets['home_or_away']['probability']:.1%}"
        )
        if "over_1_5_goals" in markets:
            st.caption(
                f"Over 1.5 goals: {markets['over_1_5_goals']['probability']:.1%} · "
                f"Under 1.5: {markets['under_1_5_goals']['probability']:.1%}"
            )
        else:
            st.caption("Totals probability unavailable: no Over/Under odds were supplied.")
        if report["slump_teams"]:
            st.warning(
                "Recent-form heuristic flagged " + ", ".join(report["slump_teams"])
                + ". Adjusted probabilities are illustrative only."
            )
            for outcome, label in (
                ("home", report["home_team"]),
                ("draw", "Draw"),
                ("away", report["away_team"]),
            ):
                st.write(f"{label}, form-adjusted: {report['form_adjusted_probabilities'][outcome]:.1%}")
        if report["review_required"]:
            st.warning("Review market: " + " ".join(report["review_flags"]))


client = create_client(SUPABASE_URL, SUPABASE_KEY)
user = current_user(client)
if user is None:
    render_auth(client)
    st.stop()

user_metadata = user.user_metadata or {}
registered_phone = str(user_metadata.get("sportybet_phone", ""))
phone_digits = re.sub(r"\D", "", registered_phone)
phone_hint = f"ending {phone_digits[-4:]}" if phone_digits else "not set"

header, signout = st.columns([3, 1])
with header:
    st.markdown('<div class="app-kicker">SPORTYBET INSTANT VIRTUAL</div>', unsafe_allow_html=True)
    st.title("Deep analyzer")
    st.caption(f"Signed in as {user.email or 'account'} · Profile phone {phone_hint}")
with signout:
    if st.button("Sign out", icon=":material/logout:"):
        try:
            client.auth.sign_out()
        except Exception:
            pass
        st.session_state.pop("auth_tokens", None)
        st.rerun()

st.caption(
    "Predictions use odds-derived probabilities and generic heuristics, not a verified SportyBet model. "
    "No bet is placed and no result is guaranteed."
)

if "mock_fixtures" not in st.session_state:
    st.session_state.mock_fixtures = generate_mock_fixtures(6)
if "team_form" not in st.session_state:
    st.session_state.team_form = {}
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.subheader("Screenshot analyzer")
upload = st.file_uploader(
    "Upload SportyBet Instant Virtual Screenshot",
    type=["png", "jpg", "jpeg", "webp"],
    help="Use a clear screenshot where each fixture row contains two 3-letter team codes and three 1X2 odds.",
)

upload_hash = hashlib.sha256(upload.getvalue()).hexdigest() if upload else None
if upload_hash != st.session_state.get("upload_hash"):
    st.session_state.upload_hash = upload_hash
    st.session_state.pop("parsed_upload", None)
    st.session_state.pop("upload_reports", None)

if upload:
    st.image(upload.getvalue(), caption="Selected screenshot", alt="SportyBet Instant Virtual screenshot")
    if st.button("Read screenshot", type="primary", icon=":material/document_scanner:"):
        try:
            with st.spinner("Reading screenshot on CPU..."):
                st.session_state.parsed_upload = extract_matches(upload.getvalue())
        except ScreenshotProcessingError as error:
            st.error(str(error))
        except Exception:
            st.error("OCR failed. Check the EasyOCR model download and try a clearer screenshot.")
        st.rerun()

parsed = st.session_state.get("parsed_upload")
if parsed:
    if not parsed["fixtures"]:
        st.warning("No complete match rows were recognized. Verify the screenshot and try again.")
        with st.expander("OCR text for review"):
            st.text(parsed["ocr_text"] or "No text recognized.")
    else:
        st.caption(parsed["parsing_note"])
        teams = list(
            dict.fromkeys(
                team
                for fixture in parsed["fixtures"]
                for team in (fixture["home_team"], fixture["away_team"])
            )
        )
        with st.form("analyze_screenshot_form"):
            st.markdown("**Optional recent form**")
            st.caption("Enter up to five W, D, or L results per code. This heuristic is not historically calibrated.")
            form_inputs = {
                team: st.text_input(
                    f"Recent form for {team}",
                    key=f"form_{team}",
                    max_chars=10,
                    placeholder="e.g. WDLWL",
                )
                for team in teams
            }
            submitted = st.form_submit_button(
                "Calculate probabilities", type="primary", icon=":material/analytics:"
            )
        if submitted:
            invalid = [team for team, form in form_inputs.items() if form and not re.fullmatch(r"[WDL\s,]+", form.upper())]
            if invalid:
                st.error("Recent form accepts only W, D, and L characters.")
            else:
                normalized_form = {
                    team: re.sub(r"[\s,]", "", form.upper())
                    for team, form in form_inputs.items()
                }
                predictor = InstantVirtualPredictor()
                st.session_state.upload_reports = predictor.analyze_matches(
                    parsed["fixtures"], normalized_form
                )
                st.session_state.analysis_history.insert(
                    0,
                    {
                        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        "fixture_count": len(parsed["fixtures"]),
                        "source": "screenshot",
                    },
                )
                st.session_state.analysis_history = st.session_state.analysis_history[:20]

if st.session_state.get("upload_reports"):
    st.header("Screenshot market estimates")
    for prediction in st.session_state.upload_reports:
        render_prediction(prediction)

st.divider()
st.subheader("Mock Instant Virtual feed")
st.caption("Simulated fallback data for local testing. It is not a live SportyBet feed.")

if st.button("Generate a new mock round", icon=":material/refresh:"):
    st.session_state.mock_fixtures = generate_mock_fixtures(6, seed=random.SystemRandom().randrange(1, 2**31))
    st.rerun()

mock_fixtures = st.session_state.mock_fixtures
mock_reports = InstantVirtualPredictor().analyze_matches(mock_fixtures, st.session_state.team_form)
for prediction in mock_reports:
    render_prediction(prediction)

with st.expander("Settle mock round and update session form"):
    st.caption("This creates random demo scores and updates W/D/L form in this browser session only.")
    if st.button("Settle mock results", key="settle_mock_round"):
        rng = random.Random()
        settled = [settle_mock_fixture(fixture, rng) for fixture in mock_fixtures]
        for result in settled:
            home = result["home_team"]
            away = result["away_team"]
            outcome = result["outcome"]
            home_result = "W" if outcome == "home" else "L" if outcome == "away" else "D"
            away_result = "W" if outcome == "away" else "L" if outcome == "home" else "D"
            st.session_state.team_form[home] = (st.session_state.team_form.get(home, "") + home_result)[-5:]
            st.session_state.team_form[away] = (st.session_state.team_form.get(away, "") + away_result)[-5:]
        st.session_state.mock_results = settled
        st.session_state.analysis_history.insert(
            0,
            {
                "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "fixture_count": len(settled),
                "source": "mock round",
            },
        )
        st.session_state.analysis_history = st.session_state.analysis_history[:20]
        st.rerun()

if st.session_state.get("mock_results"):
    st.markdown("**Latest simulated scores**")
    results_table = [
        {
            "Match": f"{result['home_team']} vs {result['away_team']}",
            "Score": result["score"],
            "Outcome": result["outcome"].upper(),
        }
        for result in st.session_state.mock_results
    ]
    st.dataframe(pd.DataFrame(results_table), hide_index=True)

if st.session_state.analysis_history:
    st.subheader("Session activity")
    st.caption("History and team form stay in Streamlit session state and are not stored in Supabase tables.")
    st.dataframe(pd.DataFrame(st.session_state.analysis_history), hide_index=True)