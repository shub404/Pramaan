import httpx

from src.config import LLM_MODEL, OLLAMA_BASE_URL
from src.models.schemas import EvidenceSource

_SYSTEM_PROMPT = (
    "You are a factual verification analyst. You will be given a claim, a verdict label, "
    "and a set of evidence snippets collected from web sources. "
    "Write a single, direct analytical paragraph that explains why the evidence leads to the given verdict. "
    "Reference specific data points or statements from the provided evidence snippets. "
    "If the verdict is CONTRADICTORY, explicitly describe the conflicting accounts found across sources. "
    "If the verdict is UNVERIFIABLE, state clearly what information was absent or insufficient. "
    "Do not use introductory phrases, pleasantries, transitional filler, or any meta-commentary. "
    "Begin immediately with the factual analysis."
)


async def generate_explanation(
    claim_text: str, verdict: str, sources: list[EvidenceSource]
) -> str:
    evidence_block = "\n".join(
        f"[{i + 1}] {src.url}\n{src.snippet_text}"
        for i, src in enumerate(sources)
    )

    user_content = (
        f"Claim: {claim_text}\n\n"
        f"Verdict: {verdict}\n\n"
        f"Evidence:\n{evidence_block or 'No evidence snippets were retrieved for this claim.'}"
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except (httpx.HTTPStatusError, httpx.TimeoutException, KeyError, ValueError):
            return f"Explanation generation failed. Verdict: {verdict}."
