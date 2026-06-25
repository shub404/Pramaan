import json
import time

import requests
import streamlit as st

_BACKEND = "http://localhost:8000"
_TERMINAL_STATES = {"VERIFIED", "FAILED", "FAILED_TIMEOUT", "NO_VERIFIABLE_CLAIMS_FOUND"}
_TIMEOUT_SECONDS = 120


def _parse_summary(raw: str) -> tuple[str, list[dict]]:
    try:
        data = json.loads(raw)
        return data.get("explanation", raw), data.get("sources", [])
    except (json.JSONDecodeError, TypeError):
        return raw or "", []


def _render_verdict_badge(verdict: str):
    if verdict == "SUPPORTED":
        st.success(verdict)
    elif verdict == "REFUTED":
        st.error(verdict)
    elif verdict == "CONTRADICTORY":
        st.warning(verdict)
    else:
        st.info(verdict or "UNVERIFIABLE")


def _render_claim_card(claim: dict):
    with st.container(border=True):
        st.markdown(f"**{claim.get('claim_text', '')}**")
        col_verdict, col_conf = st.columns([2, 1])
        with col_verdict:
            _render_verdict_badge(claim.get("verdict_label", "UNVERIFIABLE"))
        with col_conf:
            score = claim.get("composite_confidence_score") or 0.0
            st.metric("Confidence", f"{score * 100:.0f}%")
        explanation, sources = _parse_summary(claim.get("explanation_summary") or "")
        st.write(explanation)
        with st.expander("Evidence Sources"):
            if sources:
                for src in sources:
                    st.markdown(f"**{src.get('url', '')}**")
                    st.caption(src.get("snippet_text", ""))
            else:
                st.write("No evidence sources were retained for this claim.")


st.set_page_config(page_title="Pramaan | LiveFact AI", layout="centered")
st.title("Pramaan | LiveFact AI Engine")
st.caption("Local AI-driven fact-verification for YouTube videos.")

url_input = st.text_input(
    "YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=...",
)
run_button = st.button("Run Verification", type="primary")

if run_button and url_input:
    try:
        resp = requests.post(
            f"{_BACKEND}/api/verify",
            json={"url": url_input},
            timeout=10,
        )
        resp.raise_for_status()
        st.session_state.session_uuid = resp.json()["session_uuid"]
        st.session_state.start_time = time.time()
        st.session_state.results = None
        st.session_state.timed_out = False
    except requests.RequestException as exc:
        st.error(f"Could not reach the Pramaan backend: {exc}")
        st.stop()

if (
    "session_uuid" in st.session_state
    and st.session_state.get("results") is None
    and not st.session_state.get("timed_out", False)
):
    session_uuid = st.session_state.session_uuid
    start_time = st.session_state.start_time
    status_area = st.empty()

    while True:
        if time.time() - start_time >= _TIMEOUT_SECONDS:
            st.session_state.timed_out = True
            status_area.warning(
                "Verification timed out after 120 seconds. "
                "The backend may still be processing in the background."
            )
            break

        try:
            poll_resp = requests.get(
                f"{_BACKEND}/api/status/{session_uuid}", timeout=10
            )
            poll_resp.raise_for_status()
            claims = poll_resp.json().get("claims", [])
        except requests.RequestException:
            time.sleep(2)
            continue

        with status_area.container():
            if not claims:
                st.info("Ingesting transcript and extracting claims...")
            else:
                for claim in claims:
                    status = claim.get("verification_status", "PENDING")
                    if status in _TERMINAL_STATES:
                        _render_claim_card(claim)
                    else:
                        st.info(
                            f"{claim.get('claim_text', 'Claim')} — `{status}`"
                        )

        if claims and all(
            c.get("verification_status") in _TERMINAL_STATES for c in claims
        ):
            st.session_state.results = claims
            status_area.empty()
            break

        time.sleep(2)

if st.session_state.get("results"):
    st.subheader("Verification Report")
    for claim in st.session_state.results:
        _render_claim_card(claim)
