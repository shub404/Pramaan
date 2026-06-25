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
    update_claim_status,
)
from src.pipeline.analyzer import evaluate_fact
from src.pipeline.extractor import extract_claims_from_chunks
from src.pipeline.generator import generate_explanation
from src.pipeline.retriever import get_relevant_evidence
from src.pipeline.transcriber import get_overlapping_chunks

_logger = logging.getLogger(__name__)

_YOUTUBE_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=[\w-]{11}|youtu\.be/[\w-]{11})"
)


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
        chunks = await get_overlapping_chunks(video_url)
        claims = await extract_claims_from_chunks(chunks, video_url)
        await insert_claims(claims, session_uuid)
        inserted_claims = claims

        for claim in claims:
            await update_claim_status(claim.claim_id, "RETRIEVING_EVIDENCE")
            evidence_items = await get_relevant_evidence(claim.claim_text)
            await update_claim_status(claim.claim_id, "RUNNING_VERIFICATION")
            result = await evaluate_fact(claim, evidence_items)
            explanation_text = await generate_explanation(
                claim.claim_text,
                result["verdict_label"],
                result["evidence_sources"],
            )
            summary = json.dumps({
                "explanation": explanation_text,
                "sources": [
                    {"url": s.url, "snippet_text": s.snippet_text}
                    for s in result["evidence_sources"]
                ],
            })
            await finalize_claim_verdict(
                claim_id=claim.claim_id,
                confidence=result["composite_confidence_score"],
                label=result["verdict_label"],
                summary=summary,
            )

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
    claims = await get_session_status(session_uuid)
    return {"session_uuid": session_uuid, "claims": claims}
