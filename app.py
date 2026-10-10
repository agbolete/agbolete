"""SportyBet Instant Virtual screenshot analysis dashboard."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from typing import Any
from urllib.parse import urlparse

import streamlit as st
from supabase import create_client

from image_processor import ScreenshotProcessingError, extract_matches
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
    .app-kicker { color: #c8f169; font-size: .76rem; font-weight: 700; }
    .direct-prediction {
        min-height: 28vh;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #c8f169;
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        overflow-wrap: anywhere;
    }
    @media (max-width: 520px) {
        [data-testid="stMain"] { padding: .8rem .65rem 1.5rem; }
        .direct-prediction { min-height: 24vh; font-size: 2rem; }
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
    prediction = report.get("Staking_Signal")
    if prediction in {"HOME WIN", "AWAY WIN", "DRAW"}:
        st.markdown(
            f'<div class="direct-prediction">{prediction}</div>',
            unsafe_allow_html=True,
        )


client = create_client(SUPABASE_URL, SUPABASE_KEY)
user = current_user(client)
if user is None:
    render_auth(client)
    st.stop()

header, signout = st.columns([3, 1])
with header:
    if not st.session_state.get("upload_reports"):
        st.markdown('<div class="app-kicker">SPORTYBET INSTANT VIRTUAL</div>', unsafe_allow_html=True)
        st.title("Instant Virtual Analyzer")
with signout:
    if st.button("Sign out", icon=":material/logout:"):
        try:
            client.auth.sign_out()
        except Exception:
            pass
        st.session_state.pop("auth_tokens", None)
        st.rerun()

upload = st.file_uploader(
    "Upload SportyBet Instant Virtual Screenshot",
    type=["png", "jpg", "jpeg", "webp"],
)

upload_hash = hashlib.sha256(upload.getvalue()).hexdigest() if upload else None
if upload_hash != st.session_state.get("upload_hash"):
    st.session_state.upload_hash = upload_hash
    st.session_state.pop("parsed_upload", None)
    st.session_state.pop("upload_reports", None)

if upload:
    if st.button("Analyze screenshot", type="primary", icon=":material/document_scanner:"):
        try:
            with st.spinner(" "):
                parsed = extract_matches(upload.getvalue())
                st.session_state.upload_reports = InstantVirtualPredictor().analyze_matches(
                    parsed["fixtures"]
                )
                if not parsed["fixtures"]:
                    st.session_state.upload_reports = []
        except ScreenshotProcessingError:
            st.session_state.upload_reports = []
        except Exception:
            st.session_state.upload_reports = []
        st.rerun()

if st.session_state.get("upload_reports"):
    for prediction in st.session_state.upload_reports:
        render_prediction(prediction)
