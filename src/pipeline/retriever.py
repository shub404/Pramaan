import asyncio
import logging
import random
import re
import time

import faiss
import numpy as np
from ddgs import DDGS

from src.config import (
    COSINE_SIMILARITY_THRESHOLD,
    DDG_MAX_RESULTS,
    EVIDENCE_TOP_K,
    POLITICAL_COSINE_THRESHOLD,
    POLITICAL_DDG_MAX_RESULTS,
)
from src.pipeline._model_cache import get_embedding_model
from src.pipeline._ollama_client import call_ollama

_logger = logging.getLogger(__name__)
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")
_MAX_RETRIES = 3

_QUERY_REFORM_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

_QUERY_REFORM_SYSTEM_PROMPT = (
    "Convert this claim into a precise 5-8 word web search query targeting named entities, "
    "specific figures, and key events. Maximize search specificity. Return ONLY valid JSON."
)

_QUERY_BACKUP_SYSTEM_PROMPT = (
    "Reframe this claim as an alternative 5-7 word search query using different keywords — "
    "synonyms, related entities, or a key sub-fact. Return ONLY valid JSON."
)


def _split_into_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _run_ddg_search(query: str) -> list[dict]:
    for attempt in range(_MAX_RETRIES):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=DDG_MAX_RESULTS))
            return [
                {"title": r.get("title", ""), "snippet": r.get("body", ""), "href": r.get("href", "")}
                for r in results if r.get("body") and r.get("href")
            ]
        except Exception as exc:
            delay = (2 ** attempt) + random.uniform(0, 1)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
            else:
                _logger.warning("DDG search failed after %d attempts for '%s': %s", _MAX_RETRIES, query, exc)
    return []


def _run_ddg_targeted(query: str, use_news: bool, max_results: int) -> list[dict]:
    """Sync DDG search supporting both text and news modes with unified output format."""
    for attempt in range(_MAX_RETRIES):
        try:
            with DDGS() as ddgs:
                if use_news:
                    raw = list(ddgs.news(query, max_results=max_results))
                    return [
                        {"title": r.get("title", ""), "snippet": r.get("body", ""), "href": r.get("url", "")}
                        for r in raw if r.get("body") and r.get("url")
                    ]
                else:
                    raw = list(ddgs.text(query, max_results=max_results))
                    return [
                        {"title": r.get("title", ""), "snippet": r.get("body", ""), "href": r.get("href", "")}
                        for r in raw if r.get("body") and r.get("href")
                    ]
        except Exception as exc:
            delay = (2 ** attempt) + random.uniform(0, 1)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
            else:
                _logger.warning("DDG targeted search failed after %d attempts for '%s': %s", _MAX_RETRIES, query, exc)
    return []


async def fetch_search_results(query: str) -> list[dict]:
    return await asyncio.to_thread(_run_ddg_search, query)


def _faiss_rank(claim_text: str, sentences: list[str], sources: list[str], threshold: float) -> list[dict]:
    """Embed sentences + claim, return top-K above threshold ranked by cosine similarity."""
    model = get_embedding_model()
    sentence_vectors = model.encode(sentences, convert_to_numpy=True).astype(np.float32)
    claim_vector = model.encode([claim_text], convert_to_numpy=True).astype(np.float32)

    faiss.normalize_L2(sentence_vectors)
    faiss.normalize_L2(claim_vector)

    index = faiss.IndexFlatIP(sentence_vectors.shape[1])
    index.add(sentence_vectors)
    scores, indices = index.search(claim_vector, k=len(sentences))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score >= threshold:
            results.append({"url": sources[idx], "snippet_text": sentences[idx]})
        if len(results) >= EVIDENCE_TOP_K:
            break
    return results


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

    return _faiss_rank(claim_text, sentences, sources, COSINE_SIMILARITY_THRESHOLD)


async def get_ddg_evidence_targeted(claim_text: str, use_news: bool = False) -> list[dict]:
    """
    Political/current_affairs evidence retrieval.
    - Qwen reformulates the DDG query for precision
    - News search mode available for current_affairs claims
    - Fires a backup query if < 3 raw results returned
    - FAISS ranks against original claim with POLITICAL_COSINE_THRESHOLD (0.50)
    """
    # 1. Reformulate primary query
    try:
        data = await call_ollama(
            messages=[
                {"role": "system", "content": _QUERY_REFORM_SYSTEM_PROMPT},
                {"role": "user", "content": claim_text},
            ],
            schema=_QUERY_REFORM_SCHEMA,
            timeout=20.0,
        )
        query = (data.get("query") or "").strip() or claim_text
    except Exception as exc:
        _logger.warning("Query reformulation failed, using raw claim: %s", exc)
        query = claim_text

    # 2. Primary search
    raw_results = await asyncio.to_thread(_run_ddg_targeted, query, use_news, POLITICAL_DDG_MAX_RESULTS)

    # 3. Backup query if < 3 results
    if len(raw_results) < 3:
        try:
            backup = await call_ollama(
                messages=[
                    {"role": "system", "content": _QUERY_BACKUP_SYSTEM_PROMPT},
                    {"role": "user", "content": claim_text},
                ],
                schema=_QUERY_REFORM_SCHEMA,
                timeout=20.0,
            )
            backup_query = (backup.get("query") or "").strip()
            if backup_query and backup_query.lower() != query.lower():
                backup_results = await asyncio.to_thread(
                    _run_ddg_targeted, backup_query, use_news, POLITICAL_DDG_MAX_RESULTS
                )
                seen = {r["href"] for r in raw_results}
                raw_results += [r for r in backup_results if r["href"] not in seen]
        except Exception as exc:
            _logger.warning("Backup query failed: %s", exc)

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

    return _faiss_rank(claim_text, sentences, sources, POLITICAL_COSINE_THRESHOLD)


async def get_evidence_broad(claim_text: str) -> list[dict]:
    """
    Last-resort retrieval for background re-verification of claims that found
    NO evidence on the first pass. Deliberately different from the first attempt:
      - uses the RAW claim (no Qwen reformulation — zero extra Ollama load)
      - unions web (text) AND news results
      - ranks at the looser POLITICAL_COSINE_THRESHOLD
    """
    text_results, news_results = await asyncio.gather(
        asyncio.to_thread(_run_ddg_targeted, claim_text, False, POLITICAL_DDG_MAX_RESULTS),
        asyncio.to_thread(_run_ddg_targeted, claim_text, True, POLITICAL_DDG_MAX_RESULTS),
    )

    seen: set[str] = set()
    merged: list[dict] = []
    for r in text_results + news_results:
        if r["href"] not in seen:
            seen.add(r["href"])
            merged.append(r)

    if not merged:
        return []

    sentences: list[str] = []
    sources: list[str] = []
    for result in merged:
        for sentence in _split_into_sentences(result["snippet"]):
            sentences.append(sentence)
            sources.append(result["href"])

    if not sentences:
        return []

    return _faiss_rank(claim_text, sentences, sources, POLITICAL_COSINE_THRESHOLD)
