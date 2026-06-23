import hashlib

import httpx
from pydantic import BaseModel, ValidationError
from sentence_transformers import util

from src.config import (
    LLM_MODEL,
    MAX_CLAIMS_PER_VIDEO,
    OLLAMA_BASE_URL,
    SEMANTIC_DEDUPLICATION_THRESHOLD,
)
from src.models.schemas import ClaimObject
from src.pipeline._model_cache import get_embedding_model


class ExtractedClaim(BaseModel):
    claim_text: str
    importance_score: int


class ExtractedClaimsList(BaseModel):
    claims: list[ExtractedClaim]


_SYSTEM_PROMPT = (
    "You are a precise fact-extraction engine. From the provided text, extract only "
    "clear, independently verifiable factual assertions. Each claim must be a complete, "
    "standalone sentence that can be verified against external sources.\n\n"
    "Do NOT extract:\n"
    "- Personal opinions, beliefs, or subjective assessments\n"
    "- Greetings, sign-offs, or pleasantries\n"
    "- Personal anecdotes or individual experiences\n"
    "- Humor, sarcasm, or rhetorical questions\n"
    "- Vague or unverifiable generalizations\n\n"
    "Assign an importance_score from 1 to 5 per claim:\n"
    "1 = Common knowledge or trivial detail\n"
    "2 = General public-interest fact\n"
    "3 = Notable scientific, economic, or social data point\n"
    "4 = Significant statistical finding or historical event\n"
    "5 = Critical scientific, historical, socioeconomic, or public-health fact\n\n"
    "Return ONLY valid JSON matching the required schema. No preamble. No explanation."
)

# backup warning prompt to return only JSON format
_STRICT_ADDENDUM = (
    "\n\nCRITICAL: Output ONLY the JSON object. Any text outside the JSON structure "
    "will cause a system failure."
)

_MAX_LLM_RETRIES = 2

# Extract the claims from the chunk of transcript provided
async def _extract_from_chunk(
    client: httpx.AsyncClient, chunk: dict
) -> list[ExtractedClaim]:
    schema = ExtractedClaimsList.model_json_schema()

    for attempt in range(_MAX_LLM_RETRIES + 1):
        system_content = _SYSTEM_PROMPT if attempt == 0 else _SYSTEM_PROMPT + _STRICT_ADDENDUM
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": chunk["text"]},
            ],
            "format": schema,
            "stream": False,
        }
        try:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            parsed = ExtractedClaimsList.model_validate_json(content)
            return parsed.claims
        except (
            httpx.HTTPStatusError,
            httpx.TimeoutException,
            KeyError,
            ValidationError,
            ValueError,
        ):
            if attempt == _MAX_LLM_RETRIES:
                return []

    return []

# Remove duplicates from the claims returned by above function
def deduplicate_claims(claims: list[ClaimObject]) -> list[ClaimObject]:
    seen: dict[str, ClaimObject] = {}
    for claim in claims:
        key = " ".join(claim.claim_text.lower().split())
        existing = seen.get(key)
        if existing is None or claim.importance_score > existing.importance_score:
            seen[key] = claim

    unique = list(seen.values())

    if len(unique) < 2:
        return unique

    model = get_embedding_model()
    embeddings = model.encode([c.claim_text for c in unique], convert_to_tensor=True)
    cos_sim_matrix = util.cos_sim(embeddings, embeddings)

    absorbed = [False] * len(unique)
    result: list[ClaimObject] = []

    for i, base in enumerate(unique):
        if absorbed[i]:
            continue
        merged = base
        for j in range(i + 1, len(unique)):
            if absorbed[j]:
                continue
            if float(cos_sim_matrix[i][j]) >= SEMANTIC_DEDUPLICATION_THRESHOLD:
                candidate = unique[j]
                winner_text = (
                    candidate.claim_text
                    if candidate.importance_score > merged.importance_score
                    else merged.claim_text
                )
                merged = ClaimObject(
                    claim_id="",
                    session_uuid="",
                    claim_text=winner_text,
                    importance_score=max(merged.importance_score, candidate.importance_score),
                    timestamp_start=min(merged.timestamp_start, candidate.timestamp_start),
                    timestamp_end=max(merged.timestamp_end, candidate.timestamp_end),
                    verification_status="PENDING",
                )
                absorbed[j] = True
        result.append(merged)

    return result

# sort by importance, and remove irrelevant claims 
# also generates claim id for each
async def extract_claims_from_chunks(
    chunks: list[dict], video_url: str
) -> list[ClaimObject]:
    raw: list[ClaimObject] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        for chunk in chunks:
            extracted = await _extract_from_chunk(client, chunk)
            for ec in extracted:
                raw.append(
                    ClaimObject(
                        claim_id="",
                        session_uuid="",
                        timestamp_start=chunk["timestamp_start"],
                        timestamp_end=chunk["timestamp_end"],
                        claim_text=ec.claim_text,
                        importance_score=ec.importance_score,
                        verification_status="PENDING",
                    )
                )

    if not raw:
        return []

    deduplicated = deduplicate_claims(raw)

    # completely ignores of importance_score is 1
    # considers verifying once if the score is > 1
    if not deduplicated or all(c.importance_score == 1 for c in deduplicated):
        return []

    deduplicated.sort(key=lambda c: c.importance_score, reverse=True)
    top = deduplicated[:MAX_CLAIMS_PER_VIDEO]

    finalized: list[ClaimObject] = []
    for claim in top:
        normalized = " ".join(claim.claim_text.lower().split())
        claim_id = hashlib.sha256(
            f"{video_url}_{normalized}_{claim.timestamp_start}".encode()
        ).hexdigest()[:8]
        finalized.append(
            ClaimObject(
                claim_id=claim_id,
                session_uuid="",
                timestamp_start=claim.timestamp_start,
                timestamp_end=claim.timestamp_end,
                claim_text=claim.claim_text,
                importance_score=claim.importance_score,
                verification_status="PENDING",
            )
        )

    return finalized
