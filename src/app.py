import asyncio
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from src.config import OLLAMA_BASE_URL
from src.database import (
    create_session,
    finalize_claim_verdict,
    get_session_status,
    init_db,
    insert_claims,
    store_session_timeline,
    update_claim_status,
)
from src.pipeline.analyzer import evaluate_fact
from src.pipeline.extractor import extract_claims_from_chunks
from src.pipeline.generator import generate_explanation
from src.pipeline.retriever import get_relevant_evidence
from src.pipeline.transcriber import get_overlapping_chunks, get_raw_fragments

_logger = logging.getLogger(__name__)

_YOUTUBE_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=[\w-]{11}|youtu\.be/[\w-]{11})"
)


def _find_claim_anchor(claim_text: str, fragments: list[dict]) -> int:
    claim_words = set(claim_text.lower().split())
    best_score, best_start = 0, 0
    for f in fragments:
        overlap = len(claim_words & set(f["text"].lower().split()))
        if overlap > best_score:
            best_score, best_start = overlap, int(f["start"])
    return best_start


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama returned HTTP {response.status_code}. "
                    "Ensure the Ollama service is healthy before starting Pramaan."
                )
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
                "Start Ollama with `ollama serve` before launching this application."
            )
    await init_db()
    yield


app = FastAPI(title="Pramaan", version="0.1.0", lifespan=lifespan)


class VerifyRequest(BaseModel):
    url: str


async def _run_pipeline(session_uuid: str, video_url: str):
    inserted_claims = []
    try:
        chunks, fragments = await asyncio.gather(
            get_overlapping_chunks(video_url),
            get_raw_fragments(video_url),
        )

        timeline_map: dict[str, dict] = {}

        if fragments:
            max_second = max(int(f["start"] + f["duration"]) for f in fragments)
            for s in range(0, max_second + 1):
                timeline_map[str(s)] = {"quoted_text": "", "is_factual": False}
            for f in fragments:
                sec_start = int(f["start"])
                sec_end = int(f["start"] + f["duration"])
                for s in range(sec_start, sec_end + 1):
                    if str(s) in timeline_map:
                        timeline_map[str(s)]["quoted_text"] = f["text"]

        claims = await extract_claims_from_chunks(chunks, video_url)
        await insert_claims(claims, session_uuid)
        inserted_claims = claims

        await asyncio.gather(
            *[update_claim_status(c.claim_id, "RETRIEVING_EVIDENCE") for c in claims]
        )
        all_evidence = await asyncio.gather(
            *[get_relevant_evidence(c.claim_text) for c in claims]
        )

        await asyncio.gather(
            *[update_claim_status(c.claim_id, "RUNNING_VERIFICATION") for c in claims]
        )
        all_results = await asyncio.gather(
            *[evaluate_fact(c, e) for c, e in zip(claims, all_evidence)]
        )

        all_explanations = await asyncio.gather(
            *[
                generate_explanation(
                    c.claim_text, r["verdict_label"], r["evidence_sources"]
                )
                for c, r in zip(claims, all_results)
            ]
        )

        for claim, result, explanation_text in zip(claims, all_results, all_explanations):
            sources_payload = [
                {"url": s.url, "snippet_text": s.snippet_text}
                for s in result["evidence_sources"]
            ]
            summary = json.dumps({
                "explanation": explanation_text,
                "sources": sources_payload,
            })

            await finalize_claim_verdict(
                claim_id=claim.claim_id,
                confidence=result["composite_confidence_score"],
                label=result["verdict_label"],
                summary=summary,
            )

            new_confidence = result["composite_confidence_score"]
            anchor = _find_claim_anchor(claim.claim_text, fragments)
            for s in range(max(0, anchor - 5), anchor + 20):
                key = str(s)
                if key not in timeline_map:
                    continue
                existing = timeline_map[key]
                if (
                    not existing["is_factual"]
                    or new_confidence > existing.get("composite_confidence_score", 0.0)
                ):
                    timeline_map[key] = {
                        "quoted_text": existing.get("quoted_text", ""),
                        "is_factual": True,
                        "verdict_label": result["verdict_label"],
                        "composite_confidence_score": new_confidence,
                        "explanation": explanation_text,
                        "sources": sources_payload,
                    }

        await store_session_timeline(session_uuid, json.dumps(timeline_map))

    except Exception as exc:
        _logger.error(
            "Pipeline failed for session %s: %s", session_uuid, exc, exc_info=True
        )
        for claim in inserted_claims:
            await update_claim_status(claim.claim_id, "FAILED")


@app.post("/api/verify", status_code=202)
async def verify_video(request: VerifyRequest, background_tasks: BackgroundTasks):
    if not _YOUTUBE_PATTERN.match(request.url):
        raise HTTPException(status_code=422, detail="Invalid YouTube URL.")

    session_uuid = str(uuid.uuid4())
    await create_session(session_uuid, request.url)
    background_tasks.add_task(_run_pipeline, session_uuid, request.url)

    return {"session_uuid": session_uuid}


@app.get("/api/status/{session_uuid}")
async def get_status(session_uuid: str):
    result = await get_session_status(session_uuid)
    return {
        "session_uuid": session_uuid,
        "claims": result["claims"],
        "timeline": result["timeline"],
    }
