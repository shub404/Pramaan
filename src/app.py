import asyncio
import hashlib
import json
import logging
import re
import types
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import (
    CLAIM_EXTRACTION_OVERLAP,
    CLAIM_EXTRACTION_STEP,
    LLM_MODEL,
    MAX_CONCURRENT_VERIFICATIONS,
    OLLAMA_BASE_URL,
)
from src.database import (
    create_session,
    finalize_claim_verdict,
    get_pending_count,
    get_session_status,
    get_verified_claims,
    init_db,
    insert_single_claim,
    update_claim_status,
)
from src.pipeline.analyzer import evaluate_fact
from src.pipeline.retriever import get_relevant_evidence
from src.pipeline.transcriber import get_raw_fragments

_logger = logging.getLogger(__name__)

_YOUTUBE_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=[\w-]{11}|youtu\.be/[\w-]{11})"
)

# Ollama structured-output schema for claim extraction
_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_text": {"type": "string"},
                    "importance_score": {"type": "integer"},
                },
                "required": ["claim_text", "importance_score"],
            },
        }
    },
    "required": ["claims"],
}

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise fact-extraction engine. From the provided transcript excerpt, "
    "extract ONLY complete, independently verifiable factual assertions. "
    "Combine related transcript fragments into coherent, standalone sentences "
    "that can be checked against external sources.\n\n"
    "Do NOT extract:\n"
    "- Personal opinions, beliefs, or subjective assessments\n"
    "- Greetings, sign-offs, or conversational filler\n"
    "- Personal anecdotes or individual experiences\n"
    "- Vague or unverifiable generalizations\n\n"
    "Assign importance_score 1-5:\n"
    "1 = Common knowledge\n"
    "2 = General public-interest fact\n"
    "3 = Notable scientific, economic, or social data point\n"
    "4 = Significant statistical finding or historical event\n"
    "5 = Critical scientific, medical, or policy fact\n\n"
    "Return ONLY valid JSON. No preamble."
)

# In-memory session state per session_uuid
_session_state: dict[str, dict] = {}

# Semaphore: Ollama handles one inference at a time; no point queuing more
_extraction_sem = asyncio.Semaphore(1)
_verification_sem = asyncio.Semaphore(MAX_CONCURRENT_VERIFICATIONS)


def _make_claim_id(session_uuid: str, claim_text: str, timestamp: int) -> str:
    key = f"{session_uuid}{claim_text.strip().lower()}{timestamp}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _get_window_text(fragments: list[dict], from_sec: int, to_sec: int) -> str:
    """Join transcript fragment texts that fall within [from_sec, to_sec)."""
    in_window = [f for f in fragments if from_sec <= f["start"] < to_sec]
    return " ".join(f["text"] for f in in_window).strip()


def _build_auto_explanation(verdict: str, confidence: float, sources: list) -> str:
    pct = int(confidence * 100)
    n = len(sources)
    if not sources:
        return f"No relevant evidence found ({pct}% confidence)."
    try:
        domain = sources[0].url.split("/")[2]
        snippet = (sources[0].snippet_text or "")[:180]
    except Exception:
        domain, snippet = "", ""
    if verdict == "SUPPORTED":
        return f"Supported with {pct}% confidence across {n} source{'s' if n > 1 else ''}. {domain}: \"{snippet}\""
    if verdict == "REFUTED":
        return f"Refuted with {pct}% confidence across {n} source{'s' if n > 1 else ''}. {domain}: \"{snippet}\""
    if verdict == "CONTRADICTORY":
        return f"Mixed evidence from {n} source{'s' if n > 1 else ''} ({pct}%). Sources disagree."
    return f"Insufficient evidence to verify ({pct}%). {n} source{'s' if n > 1 else ''} checked."


async def _call_qwen_extract(window_text: str) -> list[dict]:
    """Send a transcript window to Qwen and return extracted claims."""
    async with _extraction_sem:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                            {"role": "user", "content": window_text},
                        ],
                        "format": _EXTRACTION_SCHEMA,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                data = json.loads(content)
                return data.get("claims", [])
        except Exception as exc:
            _logger.warning("Qwen extraction failed for window: %s", exc)
            return []


async def _verify_claim(session_uuid: str, sent: dict, claim_id: str):
    async with _verification_sem:
        try:
            await insert_single_claim(claim_id, session_uuid, sent["text"], sent["timestamp"])
            await update_claim_status(claim_id, "RETRIEVING_EVIDENCE")

            evidence = await get_relevant_evidence(sent["text"])

            if not evidence:
                summary = json.dumps({"explanation": "No relevant sources found.", "sources": []})
                await finalize_claim_verdict(claim_id, 0.0, "UNVERIFIABLE", summary)
                return

            await update_claim_status(claim_id, "RUNNING_VERIFICATION")

            result = await evaluate_fact(types.SimpleNamespace(claim_text=sent["text"]), evidence)

            explanation = _build_auto_explanation(
                result["verdict_label"],
                result["composite_confidence_score"],
                result["evidence_sources"],
            )
            sources_payload = [
                {"url": s.url, "snippet_text": s.snippet_text}
                for s in result["evidence_sources"]
            ]
            summary = json.dumps({"explanation": explanation, "sources": sources_payload})
            await finalize_claim_verdict(
                claim_id,
                result["composite_confidence_score"],
                result["verdict_label"],
                summary,
            )
        except Exception as exc:
            _logger.error("Verification failed for %s: %s", claim_id, exc, exc_info=True)
            await update_claim_status(claim_id, "FAILED")


async def _extract_and_queue(session_uuid: str, window_text: str, window_end: int):
    """Run Qwen on a transcript window, then queue each extracted claim for verification."""
    if not window_text.strip():
        return

    claims = await _call_qwen_extract(window_text)
    state = _session_state.get(session_uuid)
    if state is None:
        return

    for claim in claims:
        importance = claim.get("importance_score", 1)
        if importance <= 1:
            continue
        claim_text = (claim.get("claim_text") or "").strip()
        if not claim_text:
            continue

        # Text-based dedup to avoid re-verifying the same claim from overlapping windows
        norm = " ".join(claim_text.lower().split())
        if norm in state["seen_claim_texts"]:
            continue
        state["seen_claim_texts"].add(norm)

        sent = {"text": claim_text, "timestamp": window_end}
        claim_id = _make_claim_id(session_uuid, claim_text, window_end)
        if claim_id not in state["queued_claim_ids"]:
            state["queued_claim_ids"].add(claim_id)
            asyncio.create_task(_verify_claim(session_uuid, sent, claim_id))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama returned HTTP {resp.status_code}. "
                    "Ensure `ollama serve` is running before starting Pramaan."
                )
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Run `ollama serve` first."
            )
    await init_db()
    yield


app = FastAPI(title="Pramaan", version="2.0.0", lifespan=lifespan)


class VerifyRequest(BaseModel):
    url: str


@app.post("/api/verify", status_code=202)
async def verify_video(request: VerifyRequest):
    if not _YOUTUBE_PATTERN.match(request.url):
        raise HTTPException(status_code=422, detail="Invalid YouTube URL.")

    session_uuid = str(uuid.uuid4())
    fragments = await get_raw_fragments(request.url)

    if not fragments:
        raise HTTPException(status_code=422, detail="Could not fetch transcript for this video.")

    duration = max(int(f["start"] + f["duration"]) for f in fragments)
    await create_session(session_uuid, request.url, duration)

    _session_state[session_uuid] = {
        "fragments": fragments,
        "processed_until": 0,
        "last_extracted_until": 0,
        "queued_claim_ids": set(),
        "returned_claim_ids": set(),
        "seen_claim_texts": set(),
    }

    return {"session_uuid": session_uuid, "duration_seconds": duration}


@app.get("/api/tick/{session_uuid}")
async def tick(session_uuid: str, to_second: int = 0):
    if session_uuid not in _session_state:
        raise HTTPException(status_code=404, detail="Session not found.")

    state = _session_state[session_uuid]
    state["processed_until"] = max(state["processed_until"], to_second)

    # Trigger Qwen extraction every CLAIM_EXTRACTION_STEP seconds of new content
    if to_second >= state["last_extracted_until"] + CLAIM_EXTRACTION_STEP:
        window_start = max(0, state["last_extracted_until"] - CLAIM_EXTRACTION_OVERLAP)
        window_text = _get_window_text(state["fragments"], window_start, to_second)
        asyncio.create_task(_extract_and_queue(session_uuid, window_text, to_second))
        state["last_extracted_until"] = to_second

    all_verified = await get_verified_claims(session_uuid)
    new_claims = [c for c in all_verified if c["claim_id"] not in state["returned_claim_ids"]]
    for c in new_claims:
        state["returned_claim_ids"].add(c["claim_id"])

    pending = await get_pending_count(session_uuid)

    return {
        "new_claims": new_claims,
        "processed_until": state["processed_until"],
        "pending_count": pending,
    }


@app.get("/api/status/{session_uuid}")
async def get_status(session_uuid: str):
    result = await get_session_status(session_uuid)
    return {
        "session_uuid": session_uuid,
        "claims": result["claims"],
        "timeline": result["timeline"],
    }
