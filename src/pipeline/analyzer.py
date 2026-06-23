import asyncio
from urllib.parse import urlparse

import torch
import torch.nn.functional as F

from src.config import DOMAIN_TIER_WEIGHTS
from src.models.schemas import ClaimObject, EvidenceSource
from src.pipeline._model_cache import get_nli_components

_TIER4_DOMAINS = {"reddit.com", "twitter.com", "x.com"}
_TIER1_SUFFIXES = {"gov", "edu", "int"}
_TIER2_DOMAINS = {"wikipedia.org", "reuters.com", "bbc.com"}

_NLI_CONTRADICTION = 0
_NLI_NEUTRAL = 1
_NLI_ENTAILMENT = 2

# assigns weight as per the importance of each domain
def _resolve_domain_tier(url: str) -> tuple[int, float]:
    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.removeprefix("www.")
    except Exception:
        return 3, 0.50

    for domain in _TIER4_DOMAINS:
        if domain in hostname:
            return 4, 0.15

    parts = hostname.rsplit(".", 1)
    tld = parts[-1] if parts else ""
    if tld in _TIER1_SUFFIXES:
        return 1, DOMAIN_TIER_WEIGHTS.get(tld, 1.0)

    for domain in _TIER2_DOMAINS:
        if domain in hostname:
            return 2, DOMAIN_TIER_WEIGHTS.get(domain, 0.85)

    return 3, 0.50

# runs the NLI model against each evidence sentence and returns probability of each
def _run_nli_batch(
    claim_text: str,
    evidence_items: list[dict],
) -> list[tuple[float, float, float]]:
    tokenizer, model = get_nli_components()
    results: list[tuple[float, float, float]] = []

    for item in evidence_items:
        inputs = tokenizer(
            claim_text,
            item["snippet_text"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)
        results.append((
            float(probs[_NLI_ENTAILMENT]),
            float(probs[_NLI_CONTRADICTION]),
            float(probs[_NLI_NEUTRAL]),
        ))

    return results

# takes the input of above function and calculates the final verdict
async def evaluate_fact(claim: ClaimObject, evidence_items: list[dict]) -> dict:
    if not evidence_items:
        return {
            "verdict_label": "UNVERIFIABLE",
            "composite_confidence_score": 0.0,
            "evidence_sources": [],
            "avg_entailment": 0.0,
            "avg_contradiction": 0.0,
            "avg_neutral": 1.0,
        }

    nli_results = await asyncio.to_thread(_run_nli_batch, claim.claim_text, evidence_items)

    evidence_sources: list[EvidenceSource] = []
    weighted_confidence_sum = 0.0
    weighted_entailment_sum = 0.0
    weighted_contradiction_sum = 0.0
    weighted_neutral_sum = 0.0
    weight_total = 0.0

    for item, (p_entailment, p_contradiction, p_neutral) in zip(evidence_items, nli_results):
        tier, weight = _resolve_domain_tier(item["url"])
        source_confidence = max(p_entailment, p_contradiction)

        evidence_sources.append(
            EvidenceSource(
                url=item["url"],
                snippet_text=item["snippet_text"],
                domain_tier=tier,
                nli_entailment_prob=p_entailment,
                nli_contradiction_prob=p_contradiction,
            )
        )

        # assigns weight to the given probability
        weighted_confidence_sum += source_confidence * weight
        weighted_entailment_sum += p_entailment * weight
        weighted_contradiction_sum += p_contradiction * weight
        weighted_neutral_sum += p_neutral * weight
        weight_total += weight
    
    # takes average of each probability
    composite_confidence_score = weighted_confidence_sum / weight_total
    avg_entailment = weighted_entailment_sum / weight_total
    avg_contradiction = weighted_contradiction_sum / weight_total
    avg_neutral = weighted_neutral_sum / weight_total

    # declares final verdict!
    if avg_entailment >= 0.60 and avg_entailment > avg_contradiction:
        verdict_label = "SUPPORTED"
    elif avg_contradiction >= 0.60 and avg_contradiction > avg_entailment:
        verdict_label = "REFUTED"
    elif abs(avg_entailment - avg_contradiction) <= 0.15 and avg_entailment > 0.35 and avg_contradiction > 0.35:
        verdict_label = "CONTRADICTORY"
    else:
        verdict_label = "UNVERIFIABLE"

    return {
        "verdict_label": verdict_label,
        "composite_confidence_score": composite_confidence_score,
        "evidence_sources": evidence_sources,
        "avg_entailment": avg_entailment,
        "avg_contradiction": avg_contradiction,
        "avg_neutral": avg_neutral,
    }
