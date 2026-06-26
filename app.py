import json
import time

import requests
import streamlit as st

_BACKEND = "http://localhost:8000"
_TERMINAL_STATES = {"VERIFIED", "FAILED", "FAILED_TIMEOUT", "NO_VERIFIABLE_CLAIMS_FOUND"}
_TIMEOUT_SECONDS = 120
_SEGMENT_WINDOW = 8

st.set_page_config(page_title="Pramaan | LiveFact AI", layout="wide")

if "clock_pointer" not in st.session_state:
    st.session_state.clock_pointer = 0
if "clock_running" not in st.session_state:
    st.session_state.clock_running = False
if "processed_map" not in st.session_state:
    st.session_state.processed_map = None
if "fact_history" not in st.session_state:
    st.session_state.fact_history = []


def _rolling_segment(processed_map: dict, pointer: int) -> str:
    texts, seen = [], set()
    for s in range(max(0, pointer - _SEGMENT_WINDOW), pointer + 1):
        t = processed_map.get(str(s), {}).get("quoted_text", "")
        if t and t not in seen:
            seen.add(t)
            texts.append(t)
    return " ".join(texts)


def _verdict_badge(verdict: str):
    if verdict == "SUPPORTED":
        st.success(f"VERDICT: {verdict}")
    elif verdict == "REFUTED":
        st.error(f"VERDICT: {verdict}")
    elif verdict == "CONTRADICTORY":
        st.warning(f"VERDICT: {verdict}")
    else:
        st.info(f"VERDICT: {verdict or 'UNVERIFIABLE'}")


def _render_fact_card(entry: dict):
    with st.container(border=True):
        ts = entry.get("timestamp", 0)
        st.caption(f"⏱ Detected at {ts}s")
        quoted = entry.get("quoted_text", "")
        if quoted:
            st.markdown(f"> {quoted}")
        _verdict_badge(entry.get("verdict_label", "UNVERIFIABLE"))
        confidence = entry.get("composite_confidence_score", 0.0)
        st.metric("Confidence", f"{confidence * 100:.0f}%")
        explanation = entry.get("explanation", "")
        if explanation:
            st.write(explanation)
        sources = entry.get("sources", [])
        if sources:
            with st.expander("Evidence Sources"):
                for src in sources:
                    st.markdown(f"**{src.get('url', '')}**")
                    st.caption(src.get("snippet_text", ""))


with st.sidebar:
    st.title("Playback Controls")

    if st.session_state.clock_running:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.clock_running = False
            st.rerun()
    else:
        if st.button("▶ Resume", use_container_width=True):
            if st.session_state.processed_map:
                max_sec = max(int(k) for k in st.session_state.processed_map)
                if st.session_state.clock_pointer >= max_sec:
                    st.session_state.clock_pointer = 0
            st.session_state.clock_running = True
            st.rerun()

    col_back, col_fwd = st.columns(2)
    with col_back:
        if st.button("-2s", use_container_width=True):
            st.session_state.clock_pointer = max(0, st.session_state.clock_pointer - 2)
            st.rerun()
    with col_fwd:
        if st.button("+2s", use_container_width=True):
            st.session_state.clock_pointer += 2
            st.rerun()

    if st.session_state.processed_map:
        max_sec = max(int(k) for k in st.session_state.processed_map)
        st.progress(
            min(st.session_state.clock_pointer / max(max_sec, 1), 1.0),
            text=f"{st.session_state.clock_pointer}s / {max_sec}s",
        )

    st.divider()
    if st.button("Reset", use_container_width=True):
        st.session_state.clock_pointer = 0
        st.session_state.clock_running = False
        st.session_state.processed_map = None
        st.session_state.fact_history = []
        st.rerun()

st.title("Pramaan | LiveFact AI Engine")
st.caption("Local AI-driven fact-verification for YouTube videos.")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Pipeline Control")
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
            st.session_state.pipeline_done = False
            st.session_state.clock_pointer = 0
            st.session_state.clock_running = False
            st.session_state.processed_map = None
            st.session_state.fact_history = []
        except requests.RequestException as exc:
            st.error(f"Could not reach the Pramaan backend: {exc}")
            st.stop()

    if (
        "session_uuid" in st.session_state
        and not st.session_state.get("pipeline_done", False)
        and st.session_state.processed_map is None
    ):
        session_uuid = st.session_state.session_uuid
        start_time = st.session_state.get("start_time", time.time())
        status_area = st.empty()

        while True:
            if time.time() - start_time >= _TIMEOUT_SECONDS:
                status_area.warning(
                    "Verification timed out after 120 seconds. "
                    "The backend may still be processing in the background."
                )
                st.session_state.pipeline_done = True
                break

            try:
                poll_resp = requests.get(
                    f"{_BACKEND}/api/status/{session_uuid}", timeout=10
                )
                poll_resp.raise_for_status()
                payload = poll_resp.json()
                claims = payload.get("claims", [])
                timeline_raw = payload.get("timeline")
            except requests.RequestException:
                time.sleep(2)
                continue

            with status_area.container():
                if not claims:
                    st.info("Ingesting transcript and extracting claims...")
                else:
                    for claim in claims:
                        status = claim.get("verification_status", "PENDING")
                        label = claim.get("claim_text", "Claim")
                        if status in _TERMINAL_STATES:
                            st.success(f"{label} — `{status}`")
                        else:
                            st.info(f"{label} — `{status}`")

            all_terminal = bool(claims) and all(
                c.get("verification_status") in _TERMINAL_STATES for c in claims
            )

            if all_terminal and timeline_raw:
                st.session_state.processed_map = json.loads(timeline_raw)
                st.session_state.clock_running = True
                st.session_state.pipeline_done = True
                status_area.empty()
                break

            time.sleep(2)

with right_col:
    st.subheader("Live Verification Terminal")

    if st.session_state.processed_map is None:
        st.markdown("_Submit a YouTube URL on the left to begin live verification tracking._")

    else:
        processed_map = st.session_state.processed_map
        max_sec = max(int(k) for k in processed_map)
        pointer = st.session_state.clock_pointer
        frame = processed_map.get(str(pointer))

        if frame and frame.get("is_factual") and not any(
            h["explanation"] == frame["explanation"]
            for h in st.session_state.fact_history
        ):
            st.session_state.fact_history.append({"timestamp": pointer, **frame})

        segment_text = _rolling_segment(processed_map, pointer)
        status_label = "⏸ Paused" if not st.session_state.clock_running else "⏱ Live"
        st.caption(f"{status_label} — {pointer}s")

        if segment_text:
            st.markdown(
                f"<div style='background:#1e1e1e;color:#d4d4d4;padding:10px 14px;"
                f"border-radius:6px;font-family:monospace;font-size:0.9rem;"
                f"line-height:1.6;margin-bottom:12px'>{segment_text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("_[no transcript at this position]_")

        if st.session_state.fact_history:
            st.markdown("**Verified Claims — Newest First**")
            for entry in reversed(st.session_state.fact_history):
                _render_fact_card(entry)
        else:
            st.caption("No verified claims detected yet.")

        if st.session_state.clock_running:
            if pointer < max_sec:
                st.session_state.clock_pointer += 1
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.clock_running = False
                st.rerun()
