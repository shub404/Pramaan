import asyncio
import logging
import random
import re
import time

import faiss
import numpy as np
from ddgs import DDGS

from src.config import COSINE_SIMILARITY_THRESHOLD
from src.pipeline._model_cache import get_embedding_model

_logger = logging.getLogger(__name__)
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")
_MAX_RETRIES = 3                                    # retries 3 times if fails

# split the search snippet into sentences, as similarity check works better on individual lines
def _split_into_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

# performs DuckDuckGo search online and gathers snippets
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

# ddg search is synchronous, so we make use async wrapper
async def fetch_search_results(query: str) -> list[dict]:
    return await asyncio.to_thread(_run_ddg_search, query)

# actualy function that calls all the other functions
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

    model = get_embedding_model()

    sentence_vectors = model.encode(sentences, convert_to_numpy=True).astype(np.float32)        # embedding sentences into vectors
    claim_vector = model.encode([claim_text], convert_to_numpy=True).astype(np.float32)         # embedding claims

    # we use FAISS as it compares query vectors quickly with all other sentences instantly
    faiss.normalize_L2(sentence_vectors)            #normalizing those vectors
    faiss.normalize_L2(claim_vector)

    dim = sentence_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(sentence_vectors)

    scores, indices = index.search(claim_vector, k=len(sentences))

    raw_scores = scores[0]
    top_5 = sorted(raw_scores, reverse=True)[:5]    # considers only top 5 web results, and not other irrelevant
    print(f"  [retriever] top-5 cosine scores: {[round(float(s), 4) for s in top_5]}")
    print(f"  [retriever] threshold: {COSINE_SIMILARITY_THRESHOLD}  |  sentences above threshold: "
          f"{sum(1 for s in raw_scores if s >= COSINE_SIMILARITY_THRESHOLD)}")

    return [
        {"url": sources[idx], "snippet_text": sentences[idx]}
        for score, idx in zip(raw_scores, indices[0])
        if idx != -1 and score >= COSINE_SIMILARITY_THRESHOLD
    ]
