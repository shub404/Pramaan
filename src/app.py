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
    BACKGROUND_RETRY_BASE_DELAY,
    BACKGROUND_RETRY_MAX_ATTEMPTS,
    CLAIM_EXTRACTION_OVERLAP,
    CLAIM_EXTRACTION_STEP,
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
from src.pipeline._model_cache import get_embedding_model, get_nli_components
from src.pipeline._ollama_client import call_ollama
from src.pipeline.analyzer import evaluate_fact
from src.pipeline.retriever import (
    get_ddg_evidence_targeted,
    get_evidence_broad,
    get_relevant_evidence,
)
from src.pipeline.transcriber import get_raw_fragments
from src.pipeline.wiki_retriever import get_wikipedia_evidence

_logger = logging.getLogger(__name__)

_YOUTUBE_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=[\w-]{11}|youtu\.be/[\w-]{11})"
)

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
                    "claim_type": {
                        "type": "string",
                        "enum": ["factual", "mathematical", "visual_reference"],
                    },
                    "claim_domain": {
                        "type": "string",
                        "enum": ["science", "historical", "political", "current_affairs", "general"],
                    },
                },
                "required": ["claim_text", "importance_score", "claim_type", "claim_domain"],
            },
        }
    },
    "required": ["claims"],
}

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a fact-extraction engine for broadcast transcript verification.\n\n"
    "TASK: Extract complete, standalone, independently verifiable factual assertions. "
    "Merge related fragments into coherent claims. Discard everything else.\n\n"
    "DISCARD: opinions · predictions · rhetorical questions · anecdotes · greetings · "
    "visual descriptions ('as you can see') · vague generalizations\n\n"
    "FIELDS:\n\n"
    "claim_type — pick one:\n"
    "  factual          → real-world fact checkable externally      "
    "e.g. \"The Great Wall spans 21,196 km\"\n"
    "  mathematical     → an EXPLICIT arithmetic computation to check  "
    "e.g. \"5% of 8B is 400M\" — NOT plain statistics or quantities\n"
    "  visual_reference → requires on-screen content to verify      "
    "e.g. \"as seen in this chart\"\n\n"
    "claim_domain — pick one:\n"
    "  science          → biology · physics · chemistry · astronomy · medicine\n"
    "  historical       → events / people / facts before current decade\n"
    "  political        → government · elections · legislation · policy · geopolitics\n"
    "  current_affairs  → news or ongoing situations within last 2–3 years\n"
    "  general          → business · culture · sports · technology\n\n"
    "importance_score — integer 1–5:\n"
    "  1 common knowledge   2 general interest   3 notable data point   "
    "4 significant finding   5 critical fact\n\n"
    "RULES (strict, apply in order):\n"
    "1. Drop all visual_reference claims — do not include them in output.\n"
    "2. political/current_affairs: name the specific person, party, bill, or institution — "
    "never use \"he\", \"she\", \"they\", or \"the government\" alone.\n"
    "3. If [PRIOR CONTEXT] names a subject referenced in the current segment, "
    "use that name explicitly in the claim text.\n"
    "4. Use ONLY information stated in the transcript. Never invent or infer names, dates, "
    "locations, causes of death, or circumstances. If a detail is not stated, leave it out — "
    "do not guess to make a claim sound complete.\n"
    "5. Copy every number, unit, date, and proper name EXACTLY as stated. Do not convert, round, "
    "or add parenthetical conversions (e.g. never turn '3.1 crore tonnes' into '31 million kg').\n"
    "6. Minimum 12 words per claim — sentence fragments are not extractable.\n"
    "7. Return ONLY valid JSON. No preamble, no markdown."
)

_MATH_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "is_computation": {"type": "boolean"},
        "is_correct": {"type": "boolean"},
        "correct_value": {"type": "string"},
        "explanation": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["is_computation", "is_correct", "correct_value", "explanation", "confidence"],
}

_MATH_VERIFY_SYSTEM_PROMPT = (
    "You are a mathematical fact-checker. A statement is verifiable here ONLY if it contains an "
    "explicit arithmetic computation or numeric relationship you can check by calculation "
    "(e.g. a percentage of a stated total, a sum, a conversion using a given factor).\n\n"
    "is_computation: true ONLY when there is concrete arithmetic to perform. For a qualitative or "
    "empirical claim with nothing to calculate (e.g. 'orders fell by about half', 'X tonnes are "
    "consumed annually'), set is_computation=false and is_correct=false.\n"
    "is_correct: true only if is_computation is true AND the result is exact or within 1% rounding.\n"
    "correct_value: the mathematically correct answer, or empty string if is_computation is false.\n"
    "explanation: concise step-by-step computation, or why there is no computation to verify.\n"
    "confidence: 0.0–1.0 (1.0 = unambiguous arithmetic).\n"
    "Return ONLY valid JSON."
)

_session_state: dict[str, dict] = {}

# Semaphore for DDG + NLI concurrent tasks (Ollama semaphore lives in _ollama_client.py)
_verification_sem = asyncio.Semaphore(MAX_CONCURRENT_VERIFICATIONS)
# Background re-verification runs one-at-a-time so it never competes with live verification
_retry_sem = asyncio.Semaphore(1)


def _make_claim_id(session_uuid: str, claim_text: str, timestamp: int) -> str:
    key = f"{session_uuid}{claim_text.strip().lower()}{timestamp}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _get_window_text(fragments: list[dict], from_sec: int, to_sec: int) -> str:
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


async def _verify_math_claim(claim_text: str) -> dict | None:
    try:
        return await call_ollama(
            messages=[
                {"role": "system", "content": _MATH_VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Verify the math in: {claim_text}"},
            ],
            schema=_MATH_VERIFY_SCHEMA,
            timeout=25.0,
        )
    except Exception as exc:
        _logger.warning("Math verification failed: %s", exc)
        return None


async def _call_qwen_extract(window_text: str, prior_context: str = "") -> list[dict]:
    if prior_context:
        user_content = (
            f"[PRIOR CONTEXT — resolve references only, do not re-extract]:\n{prior_context}\n\n"
            f"[CURRENT SEGMENT]:\n{window_text}"
        )
    else:
        user_content = window_text

    try:
        data = await call_ollama(
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            schema=_EXTRACTION_SCHEMA,
            timeout=60.0,
        )
        return data.get("claims", [])
    except Exception as exc:
        _logger.warning("Qwen extraction failed for window: %s", exc)
        return []


async def _verify_claim(session_uuid: str, sent: dict, claim_id: str):
    async with _verification_sem:
        try:
            await insert_single_claim(claim_id, session_uuid, sent["text"], sent["timestamp"])
            claim_type = sent.get("claim_type", "factual")

            # --- Mathematical claims: Qwen computation, no web search ---
            # Only commit a math verdict when there is genuine arithmetic to verify. If the
            # extractor mislabeled a qualitative/empirical statement as math, fall through to
            # evidence-based verification instead of forcing a (false) REFUTED.
            if claim_type == "mathematical":
                await update_claim_status(claim_id, "RUNNING_VERIFICATION")
                math_result = await _verify_math_claim(sent["text"])
                if math_result and math_result.get("is_computation"):
                    is_correct = bool(math_result["is_correct"])
                    verdict = "SUPPORTED" if is_correct else "REFUTED"
                    correct_val = math_result.get("correct_value", "")
                    prefix = "Mathematically correct" if is_correct else "Mathematically incorrect"
                    explanation_text = f"{prefix}. {math_result.get('explanation', '')}".strip()
                    if correct_val:
                        explanation_text += f" Correct value: {correct_val}."
                    summary = json.dumps({
                        "explanation": explanation_text,
                        "sources": [{"url": "AI mathematical computation", "snippet_text": correct_val}],
                    })
                    await finalize_claim_verdict(
                        claim_id,
                        max(0.0, min(1.0, float(math_result.get("confidence", 0.85)))),
                        verdict,
                        summary,
                    )
                    return
                _logger.debug("Math claim has no computation, treating as factual: %s", sent["text"][:60])
                # fall through to evidence-based verification below

            # --- Factual claims: domain-routed evidence retrieval → NLI ---
            await update_claim_status(claim_id, "RETRIEVING_EVIDENCE")
            evidence = await _route_and_retrieve(sent)

            if not evidence:
                # Nothing found on the first pass — keep trying off the critical path.
                summary = json.dumps({
                    "explanation": "No relevant sources found yet — still checking in the background.",
                    "sources": [],
                })
                await finalize_claim_verdict(claim_id, 0.0, "UNVERIFIABLE", summary)
                asyncio.create_task(_background_reverify(session_uuid, sent, claim_id, attempt=1))
                return

            await update_claim_status(claim_id, "RUNNING_VERIFICATION")
            await _evaluate_and_finalize(claim_id, sent, evidence)
        except Exception as exc:
            _logger.error("Verification failed for %s: %s", claim_id, exc, exc_info=True)
            await update_claim_status(claim_id, "FAILED")


async def _route_and_retrieve(sent: dict) -> list[dict]:
    """Route a claim to the best evidence source based on its domain."""
    claim_domain = sent.get("claim_domain", "general")
    text = sent["text"]

    if claim_domain in ("science", "historical"):
        evidence = await get_wikipedia_evidence(text)
        if not evidence:                                  # Wikipedia miss → fall back to DDG
            evidence = await get_relevant_evidence(text)
        return evidence

    if claim_domain in ("political", "current_affairs"):
        return await get_ddg_evidence_targeted(text, use_news=(claim_domain == "current_affairs"))

    return await get_relevant_evidence(text)


async def _evaluate_and_finalize(claim_id: str, sent: dict, evidence: list[dict]) -> str:
    """
    Run NLI/Qwen adjudication on retrieved evidence and store the verdict.
    Returns the final verdict label.

    If evidence was retrieved but the engines cannot confirm or refute the claim,
    the verdict is recorded as INCONCLUSIVE (not a flat UNVERIFIABLE) so the UI can
    surface the context that WAS found — without implying a true/false judgement.
    """
    claim_domain = sent.get("claim_domain", "general")
    result = await evaluate_fact(
        types.SimpleNamespace(claim_text=sent["text"]),
        evidence,
        claim_domain=claim_domain,
    )

    verdict = result["verdict_label"]
    confidence = result["composite_confidence_score"]
    sources_payload = [
        {"url": s.url, "snippet_text": s.snippet_text}
        for s in result["evidence_sources"]
    ]

    if verdict == "UNVERIFIABLE" and sources_payload:
        verdict = "INCONCLUSIVE"
        explanation = (
            "Related context was found, but it does not clearly confirm or refute this "
            "specific claim. See the retrieved sources below."
        )
    else:
        explanation = result.get("rational_explanation") or _build_auto_explanation(
            verdict, confidence, result["evidence_sources"]
        )

    summary = json.dumps({"explanation": explanation, "sources": sources_payload})
    await finalize_claim_verdict(claim_id, confidence, verdict, summary)
    return verdict


async def _background_reverify(session_uuid: str, sent: dict, claim_id: str, attempt: int):
    """
    Re-verify a claim that found NO evidence on its first pass. Runs off the critical
    path with linear backoff. Uses a broadened retrieval (raw claim, web + news union,
    looser threshold). May only PROMOTE the verdict — a confident result or INCONCLUSIVE
    with evidence is strictly more informative than 'no sources found'.
    """
    await asyncio.sleep(BACKGROUND_RETRY_BASE_DELAY * attempt)

    if _session_state.get(session_uuid) is None:
        return  # session was reset / no longer active

    promoted = False
    async with _retry_sem:
        try:
            evidence = await get_evidence_broad(sent["text"])
            if evidence:
                verdict = await _evaluate_and_finalize(claim_id, sent, evidence)
                _logger.info("Background retry %d for %s → %s", attempt, claim_id, verdict)
                promoted = True
        except Exception as exc:
            _logger.warning("Background retry %d failed for %s: %s", attempt, claim_id, exc)

    if not promoted and attempt < BACKGROUND_RETRY_MAX_ATTEMPTS:
        asyncio.create_task(_background_reverify(session_uuid, sent, claim_id, attempt + 1))


async def _warmup(session_uuid: str):
    state = _session_state.get(session_uuid)
    if state is None:
        return
    try:
        await asyncio.gather(
            asyncio.to_thread(get_nli_components),
            asyncio.to_thread(get_embedding_model),
        )
        window_text = _get_window_text(state["fragments"], 0, CLAIM_EXTRACTION_STEP)
        if window_text:
            await _extract_and_queue(session_uuid, window_text, CLAIM_EXTRACTION_STEP)
            state["last_extracted_until"] = CLAIM_EXTRACTION_STEP
    except Exception as exc:
        _logger.error("Warmup failed for %s: %s", session_uuid, exc, exc_info=True)
    finally:
        if _session_state.get(session_uuid) is not None:
            _session_state[session_uuid]["warmup_done"] = True


async def _extract_and_queue(session_uuid: str, window_text: str, window_end: int):
    if not window_text.strip():
        return

    state = _session_state.get(session_uuid)
    if state is None:
        return

    prior_context = state.get("extraction_context", "")
    claims = await _call_qwen_extract(window_text, prior_context)

    # Update rolling context for the next window
    state["extraction_context"] = window_text[-200:].strip()

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
        claim_type = (claim.get("claim_type") or "factual").lower()
        claim_domain = (claim.get("claim_domain") or "general").lower()

        if claim_type == "visual_reference":
            _logger.debug("Skipping visual reference: %s", claim_text[:60])
            continue

        # Political claims shorter than 12 words are fragments, not verifiable assertions
        if claim_domain in ("political", "current_affairs") and len(claim_text.split()) < 12:
            _logger.debug("Skipping short political claim: %s", claim_text[:60])
            continue

        norm = " ".join(claim_text.lower().split())
        if norm in state["seen_claim_texts"]:
            continue
        state["seen_claim_texts"].add(norm)

        sent = {
            "text": claim_text,
            "timestamp": window_end,
            "claim_type": claim_type,
            "claim_domain": claim_domain,
        }
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
        "returned_claim_versions": {},
        "seen_claim_texts": set(),
        "extraction_context": "",
        "warmup_done": False,
    }
    asyncio.create_task(_warmup(session_uuid))

    return {"session_uuid": session_uuid, "duration_seconds": duration}


@app.get("/api/tick/{session_uuid}")
async def tick(session_uuid: str, to_second: int = 0):
    if session_uuid not in _session_state:
        raise HTTPException(status_code=404, detail="Session not found.")

    state = _session_state[session_uuid]
    state["processed_until"] = max(state["processed_until"], to_second)

    if to_second >= state["last_extracted_until"] + CLAIM_EXTRACTION_STEP:
        window_start = max(0, state["last_extracted_until"] - CLAIM_EXTRACTION_OVERLAP)
        window_text = _get_window_text(state["fragments"], window_start, to_second)
        asyncio.create_task(_extract_and_queue(session_uuid, window_text, to_second))
        state["last_extracted_until"] = to_second

    # Emit a claim when first verified AND again whenever a background retry bumps its
    # updated_at (so a promoted verdict reaches the UI instead of being cached forever).
    all_verified = await get_verified_claims(session_uuid)
    versions = state["returned_claim_versions"]
    new_claims = []
    for c in all_verified:
        cid = c["claim_id"]
        prev = versions.get(cid)
        if prev is None or c["updated_at"] > prev:
            new_claims.append(c)
            versions[cid] = c["updated_at"]

    pending = await get_pending_count(session_uuid)

    return {
        "new_claims": new_claims,
        "processed_until": state["processed_until"],
        "pending_count": pending,
    }


@app.get("/api/ready/{session_uuid}")
async def check_ready(session_uuid: str):
    if session_uuid not in _session_state:
        raise HTTPException(status_code=404, detail="Session not found.")
    state = _session_state[session_uuid]
    return {
        "ready": state.get("warmup_done", False),
        "pending_count": await get_pending_count(session_uuid),
    }


@app.get("/api/status/{session_uuid}")
async def get_status(session_uuid: str):
    result = await get_session_status(session_uuid)
    return {
        "session_uuid": session_uuid,
        "claims": result["claims"],
        "timeline": result["timeline"],
    }
