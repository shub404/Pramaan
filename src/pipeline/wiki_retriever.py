import logging
import re

import faiss
import httpx
import numpy as np

from src.config import EVIDENCE_TOP_K
from src.pipeline._model_cache import get_embedding_model
from src.pipeline._ollama_client import call_ollama

_logger = logging.getLogger(__name__)

_WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_HEADERS = {"User-Agent": "Pramaan/2.0 (fact-verification research tool; contact@pramaan.ai)"}

_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {"topic": {"type": "string"}},
    "required": ["topic"],
}

_ENTITY_SYSTEM_PROMPT = (
    "Extract the single most searchable Wikipedia topic from the given claim. "
    "Return only the topic name — a short noun phrase (2-5 words). "
    "No explanation. Return ONLY valid JSON."
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


async def _extract_topic(claim_text: str) -> str:
    data = await call_ollama(
        messages=[
            {"role": "system", "content": _ENTITY_SYSTEM_PROMPT},
            {"role": "user", "content": claim_text},
        ],
        schema=_ENTITY_SCHEMA,
        timeout=20.0,
    )
    return (data.get("topic") or "").strip()


async def _search_title(topic: str) -> str | None:
    async with httpx.AsyncClient(timeout=10.0, headers=_HEADERS) as client:
        resp = await client.get(
            _WIKI_SEARCH_URL,
            params={"action": "query", "list": "search", "srsearch": topic, "format": "json", "srlimit": 1},
        )
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        return results[0]["title"] if results else None


async def _fetch_summary(title: str) -> dict | None:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
    async with httpx.AsyncClient(timeout=10.0, headers=_HEADERS) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", url),
            "extract": data.get("extract", ""),
        }


async def get_wikipedia_evidence(claim_text: str) -> list[dict]:
    """
    Fetch Wikipedia evidence for a claim.
    Returns list[dict] with keys: url, snippet_text  (same contract as retriever.py)
    """
    try:
        topic = await _extract_topic(claim_text)
        if not topic:
            return []

        title = await _search_title(topic)
        if not title:
            return []

        article = await _fetch_summary(title)
        if not article or not article["extract"]:
            return []

        extract = article["extract"]
        page_url = article["url"]

        if len(extract) <= 500:
            return [{"url": page_url, "snippet_text": extract}]

        sentences = _split_sentences(extract)
        if not sentences:
            return [{"url": page_url, "snippet_text": extract[:500]}]

        model = get_embedding_model()
        sentence_vectors = model.encode(sentences, convert_to_numpy=True).astype(np.float32)
        claim_vector = model.encode([claim_text], convert_to_numpy=True).astype(np.float32)

        faiss.normalize_L2(sentence_vectors)
        faiss.normalize_L2(claim_vector)

        dim = sentence_vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(sentence_vectors)

        k = min(EVIDENCE_TOP_K, len(sentences))
        scores, indices = index.search(claim_vector, k=k)

        results = [
            {"url": page_url, "snippet_text": sentences[idx]}
            for score, idx in zip(scores[0], indices[0])
            if idx != -1 and score > 0.0
        ]
        return results or [{"url": page_url, "snippet_text": extract[:500]}]

    except Exception as exc:
        _logger.warning("Wikipedia evidence failed for '%s': %s", claim_text[:60], exc)
        return []
