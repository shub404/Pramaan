import time

import requests
import streamlit as st

_BACKEND = "http://localhost:8000"


def _fmt_time(seconds: int) -> str:
    m, s = divmod(max(0, seconds), 60)
    return f"{m:02d}:{s:02d}"


def _get_pointer() -> int:
    """Compute current clock position from wall time so it stays accurate across reruns."""
    if st.session_state.clock_running and st.session_state.clock_wall_start:
        elapsed = int(time.time() - st.session_state.clock_wall_start)
        return st.session_state.clock_pointer_at_start + elapsed
    return st.session_state.clock_pointer_at_start


def _start_clock(pointer: int | None = None):
    if pointer is not None:
        st.session_state.clock_pointer_at_start = max(0, pointer)
    st.session_state.clock_wall_start = time.time()
    st.session_state.clock_running = True


def _pause_clock():
    st.session_state.clock_pointer_at_start = _get_pointer()
    st.session_state.clock_wall_start = None
    st.session_state.clock_running = False


def _verdict_color(verdict: str) -> str:
    return {
        "SUPPORTED": "green",
        "REFUTED": "red",
        "CONTRADICTORY": "orange",
        "INCONCLUSIVE": "gray",
    }.get(verdict, "blue")


def _verdict_icon(verdict: str) -> str:
    return {
        "SUPPORTED": "✅",
        "REFUTED": "❌",
        "CONTRADICTORY": "⚠️",
        "INCONCLUSIVE": "🔍",
    }.get(verdict, "ℹ️")


def _render_claim_card(claim: dict):
    verdict = claim.get("verdict_label", "UNVERIFIABLE")
    confidence = claim.get("composite_confidence_score", 0.0)
    pct = int(confidence * 100)
    icon = _verdict_icon(verdict)
    ts = _fmt_time(claim.get("timestamp", 0))
    color = _verdict_color(verdict)

    with st.container(border=True):
        col_v, col_c, col_t = st.columns([3, 1, 1])
        with col_v:
            st.markdown(
                f"<span style='color:{color};font-weight:700;font-size:1rem'>"
                f"{icon} {verdict}</span>",
                unsafe_allow_html=True,
            )
        with col_c:
            # A truth-percentage only makes sense for an actual ruling. For INCONCLUSIVE /
            # UNVERIFIABLE a "0%" reads as "false" — so show a neutral marker instead.
            if verdict in ("SUPPORTED", "REFUTED", "CONTRADICTORY"):
                st.markdown(
                    f"<span style='font-size:1.4rem;font-weight:700;color:{color}'>{pct}%</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='font-size:1.1rem;color:{color}'>—</span>",
                    unsafe_allow_html=True,
                )
        with col_t:
            st.caption(f"⏱ {ts}")

        claim_text = claim.get("claim_text", "")
        if claim_text:
            st.markdown(f"> *{claim_text}*")

        explanation = claim.get("explanation", "")
        if explanation:
            st.write(explanation)

        sources = claim.get("sources", [])
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for src in sources:
                    url = src.get("url", "")
                    snippet = src.get("snippet_text", "")
                    if url:
                        st.markdown(f"[{url}]({url})")
                    if snippet:
                        st.caption(snippet[:200])


@st.fragment(run_every=2)
def _warmup_poll_fragment():
    """Polls /api/ready every 2s without triggering a full-page rerun (no blinking)."""
    uuid = st.session_state.get("session_uuid")
    if not uuid or st.session_state.get("warmup_done"):
        return
    try:
        resp = requests.get(f"{_BACKEND}/api/ready/{uuid}", timeout=4)
        if resp.ok and resp.json().get("ready"):
            st.session_state.warmup_done = True
            st.session_state.pending_count = resp.json().get("pending_count", 0)
            st.rerun()
    except requests.RequestException:
        pass


st.set_page_config(page_title="Pramaan | LiveFact AI", layout="wide")

# --- Session state defaults ---
for key, default in [
    ("clock_running", False),
    ("clock_wall_start", None),
    ("clock_pointer_at_start", 0),
    ("duration_seconds", 0),
    ("session_uuid", None),
    ("warmup_done", False),
    ("verified_claims", []),
    ("pending_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar ---
with st.sidebar:
    st.title("Playback Controls")

    # Only show clock controls once the pipeline is ready
    if st.session_state.warmup_done:
        if st.session_state.clock_running:
            if st.button("⏸ Pause", use_container_width=True):
                _pause_clock()
                st.rerun()
        else:
            if st.button("▶ Resume", use_container_width=True):
                _start_clock()
                st.rerun()

        col_b, col_f = st.columns(2)
        with col_b:
            if st.button("−5s", use_container_width=True):
                new_ptr = max(0, _get_pointer() - 5)
                if st.session_state.clock_running:
                    _start_clock(new_ptr)
                else:
                    st.session_state.clock_pointer_at_start = new_ptr
                st.rerun()
        with col_f:
            if st.button("+5s", use_container_width=True):
                new_ptr = min(st.session_state.duration_seconds, _get_pointer() + 5)
                if st.session_state.clock_running:
                    _start_clock(new_ptr)
                else:
                    st.session_state.clock_pointer_at_start = new_ptr
                st.rerun()

    dur = st.session_state.duration_seconds
    if dur > 0:
        pointer = _get_pointer()
        st.progress(
            min(pointer / dur, 1.0),
            text=f"{_fmt_time(pointer)} / {_fmt_time(dur)}",
        )

    st.divider()
    if st.button("Reset", use_container_width=True):
        st.session_state.clock_running = False
        st.session_state.clock_wall_start = None
        st.session_state.clock_pointer_at_start = 0
        st.session_state.duration_seconds = 0
        st.session_state.session_uuid = None
        st.session_state.warmup_done = False
        st.session_state.verified_claims = []
        st.session_state.pending_count = 0
        st.rerun()

# --- Header ---
st.title("Pramaan | LiveFact AI")
st.caption("Real-time fact verification — sentence by sentence as the video plays.")

left_col, right_col = st.columns([1, 1.6])

# --- Left column: state machine ---
with left_col:
    uuid = st.session_state.session_uuid

    if uuid is None:
        # STATE 1: Idle — show URL input
        st.subheader("Submit Video")
        url_input = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        run_btn = st.button("Load Video", type="primary", use_container_width=True)

        if run_btn and url_input:
            with st.spinner("Fetching transcript…"):
                try:
                    resp = requests.post(
                        f"{_BACKEND}/api/verify",
                        json={"url": url_input},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.session_uuid = data["session_uuid"]
                    st.session_state.duration_seconds = data["duration_seconds"]
                    st.session_state.verified_claims = []
                    st.session_state.pending_count = 0
                    st.session_state.warmup_done = False
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Could not reach Pramaan backend: {exc}")

    elif not st.session_state.warmup_done:
        # STATE 2: Warmup — fragment polls every 2s without full-page rerun (no blinking)
        st.subheader("Preparing pipeline…")
        st.info(
            "Loading AI models and running first extraction. This takes 15–30 seconds.\n\n"
            "DeBERTa NLI, MiniLM embeddings, and Qwen are being warmed up in parallel."
        )
        _warmup_poll_fragment()

    elif not st.session_state.clock_running:
        # STATE 3: Ready — wait for user to press Start
        st.subheader("Pipeline Ready")
        st.success("Models loaded. First extraction complete.")
        pending = st.session_state.pending_count
        if pending > 0:
            st.caption(f"First claim batch already queued: {pending} verifying in background.")
        st.markdown("**Start your video now, then click the button below.**")
        st.markdown("The clock will sync to your playback from second 0.")

        if st.button("▶ Start Live Verification", type="primary", use_container_width=True):
            _start_clock(0)
            st.rerun()

    else:
        # STATE 4: Running — show live stats
        pointer = _get_pointer()
        verified = len(st.session_state.verified_claims)
        pending = st.session_state.pending_count
        st.markdown(
            f"🟢 **Verified:** {verified} &nbsp;|&nbsp; "
            f"**Pending:** {pending} &nbsp;|&nbsp; "
            f"**Clock:** {_fmt_time(pointer)}"
        )

        if not st.session_state.verified_claims:
            st.info(
                "Qwen extracts claims every 15s. "
                "Results appear on the right once each claim is verified (~10–20s)."
            )

# --- Right column ---
with right_col:
    st.subheader("Verified Claims — Live Feed")

    if not st.session_state.session_uuid:
        st.markdown("_Submit a YouTube URL to start the live fact-check feed._")
    elif not st.session_state.warmup_done:
        st.markdown("_Warming up… results will appear here once verification begins._")
    elif not st.session_state.verified_claims:
        st.markdown("_No claims verified yet. Results appear here as they complete._")
    else:
        for claim in st.session_state.verified_claims:
            _render_claim_card(claim)

# --- Clock tick (runs every rerender when clock is active) ---
if st.session_state.clock_running and st.session_state.session_uuid:
    pointer = _get_pointer()
    dur = st.session_state.duration_seconds

    if pointer >= dur > 0:
        _pause_clock()
        st.rerun()
    else:
        try:
            tick_resp = requests.get(
                f"{_BACKEND}/api/tick/{st.session_state.session_uuid}",
                params={"to_second": pointer},
                timeout=5,
            )
            if tick_resp.ok:
                data = tick_resp.json()
                for claim in data.get("new_claims", []):
                    cid = claim["claim_id"]
                    # Replace any existing card for this claim (a background retry may have
                    # promoted its verdict), then surface it at the top of the feed.
                    st.session_state.verified_claims = [
                        c for c in st.session_state.verified_claims if c["claim_id"] != cid
                    ]
                    st.session_state.verified_claims.insert(0, claim)
                st.session_state.pending_count = data.get("pending_count", 0)
        except requests.RequestException:
            pass

        time.sleep(0.8)
        st.rerun()
