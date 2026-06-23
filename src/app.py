import asyncio
import re                                       # used to validate youtube url
import uuid                                     # used to generate unique ids
from contextlib import asynccontextmanager      # used to create startup & shutdown event handlers

import httpx                                    # used to make http requests to ollama
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
from src.models.schemas import ClaimObject

# making a resuable regex
_YOUTUBE_PATTERN = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=[\w-]{11}|youtu\.be/[\w-]{11})"
)

_PIPELINE_STATES = [
    "EXTRACTING_TRANSCRIPT",
    "EXTRACTING_CLAIMS",
    "RETRIEVING_EVIDENCE",
    "RUNNING_VERIFICATION",
]


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


app = FastAPI(title="Pramaan", version="0.1.0", lifespan=lifespan)  # this creates a server


class VerifyRequest(BaseModel):          # BaseModel is used to validate the url structure
    url: str

# dummy data for now, later replaced by pipeline
async def _placeholder_pipeline(session_uuid: str):
    claim_id = str(uuid.uuid4())
    placeholder = ClaimObject(
        claim_id=claim_id,
        session_uuid=session_uuid,
        timestamp_start=0,
        timestamp_end=30,
        claim_text="Placeholder claim — pipeline skeleton active.",
        importance_score=3,
        verification_status="PENDING",
    )
    await insert_claims([placeholder], session_uuid)

    for state in _PIPELINE_STATES:
        await update_claim_status(claim_id, state)
        await asyncio.sleep(2)

    await finalize_claim_verdict(
        claim_id=claim_id,
        confidence=0.0,
        label="UNVERIFIABLE",
        summary="Placeholder run — real verification pipeline not yet wired.",
    )


@app.post("/api/verify", status_code=202)
async def verify_video(request: VerifyRequest, background_tasks: BackgroundTasks):
    if not _YOUTUBE_PATTERN.match(request.url):
        raise HTTPException(status_code=422, detail="Invalid YouTube URL.")

    session_uuid = str(uuid.uuid4())            # generate unique session id
    await create_session(session_uuid, request.url)     # passs it here
    background_tasks.add_task(_placeholder_pipeline, session_uuid)      # start the pipeline

    return {"session_uuid": session_uuid}


@app.get("/api/status/{session_uuid}")
async def get_status(session_uuid: str):
    claims = await get_session_status(session_uuid)
    if not claims:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_uuid": session_uuid, "claims": claims}
