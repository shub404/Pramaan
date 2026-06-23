import asyncio
import logging
import random
import re
import time

import faiss
import numpy as np
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer

from src.config import COSINE_SIMILARITY_THRESHOLD, EMBEDDING_MODEL

_logger = logging.getLogger(__name__)
_embedding_model: SentenceTransformer | None = None
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")
_MAX_RETRIES = 3


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedding_model


def _split_into_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _run_ddg_search(query: str) -> list[dict]:
    for attempt in range(_MAX_RETRIES):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=10))
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "href": r.get("href", ""),
                }
                for r in results
                if r.get("body") and r.get("href")
            ]
        except Exception as exc:
            delay = (2 ** attempt) + random.uniform(0, 1)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
            else:
                _logger.warning(
                    "DDG search failed after %d attempts for query '%s': %s",
                    _MAX_RETRIES,
                    query,
                    exc,
                )
    return []


async def fetch_search_results(query: str) -> list[dict]:
    return await asyncio.to_thread(_run_ddg_search, query)


async def get_relevant_evidence(claim_text: str) -> list[dict]:
    raw_results = await fetch_search_results(claim_text)
    if not raw_results:
        return []

    sentences: list[str] = []
    sources: list[str] = []

    for result in raw_results:
        for sentence in _split_into_sentences(result["snippet"]):
            sentences.append(sentence)
            sources.append(result["href"])

    if not sentences:
        return []

    model = _get_embedding_model()

    sentence_vectors = model.encode(sentences, convert_to_numpy=True).astype(np.float32)
    claim_vector = model.encode([claim_text], convert_to_numpy=True).astype(np.float32)

    faiss.normalize_L2(sentence_vectors)
    faiss.normalize_L2(claim_vector)

    dim = sentence_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(sentence_vectors)

    scores, indices = index.search(claim_vector, k=len(sentences))

    return [
        {"url": sources[idx], "snippet_text": sentences[idx]}
        for score, idx in zip(scores[0], indices[0])
        if idx != -1 and score >= COSINE_SIMILARITY_THRESHOLD
    ]
