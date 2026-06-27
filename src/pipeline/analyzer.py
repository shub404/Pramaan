import asyncio
import logging
import re
from urllib.parse import urlparse

import torch
import torch.nn.functional as F

from src.config import DOMAIN_TIER_WEIGHTS
from src.models.schemas import ClaimObject, EvidenceSource
from src.pipeline._model_cache import get_nli_components
from src.pipeline._ollama_client import call_ollama

_logger = logging.getLogger(__name__)

_TIER4_DOMAINS = {"reddit.com", "twitter.com", "x.com"}
_TIER1_SUFFIXES = {"gov", "edu", "int"}
_TIER2_DOMAINS = {"wikipedia.org", "reuters.com", "bbc.com"}

_NLI_CONTRADICTION = 0
_NLI_NEUTRAL = 1
_NLI_ENTAILMENT = 2

# Below this composite confidence, DeBERTa is uncertain — ask Qwen to reason about it
_RATIONAL_THRESHOLD = 0.65

_RATIONAL_SYSTEM_PROMPT = """ROLE
You are a strict, logical fact-checking adjudicator. Your sole purpose is to evaluate a specific CLAIM against a provided set of EVIDENCE blocks.

RULES OF EVALUATION
1. BLIND JUDGE: You must base your verdict STRICTLY on the provided EVIDENCE. Do not use outside knowledge. If the EVIDENCE does not contain the information needed to verify the CLAIM, you must output UNVERIFIABLE.
2. NO MATH OR CONVERSIONS: Do not attempt to calculate, convert units, or solve equations. If a CLAIM uses miles and the EVIDENCE uses light-years, and the EVIDENCE does not explicitly state the conversion, you must output UNVERIFIABLE.
3. SEMANTIC EQUIVALENCE: Allow for reasonable linguistic variations. "Over 31 years" is logically supported by evidence stating "31.7 years". Do not refute claims based on pedantic decimal differences unless exact precision is the core subject of the claim.
4. DEFINITIONS:
   - SUPPORTED: The core facts of the CLAIM are explicitly backed by the EVIDENCE.
   - REFUTED: The EVIDENCE explicitly states a fact that makes the CLAIM impossible or false.
   - CONTRADICTORY: The EVIDENCE blocks disagree with each other.
   - UNVERIFIABLE: The EVIDENCE is empty, irrelevant, or insufficient to make a ruling.

INSTRUCTIONS
You must analyze the data using the following step-by-step logic.
Step 1: Identify the core entity and metric in the CLAIM.
Step 2: Scan the EVIDENCE for that exact entity and metric.
Step 3: Compare the two. Are they logically aligned, directly opposed, or unrelated?

OUTPUT FORMAT
You must output a strictly valid JSON object with exactly two keys. Do not output markdown blocks, backticks, or conversational text.

{"verdict": "SUPPORTED | REFUTED | CONTRADICTORY | UNVERIFIABLE", "explanation": "A concise, single-paragraph explanation of how the evidence proves or disproves the claim. Do not mention math formulas. Do not start with phrases like 'The claim is...'"}"""

_RATIONAL_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["SUPPORTED", "REFUTED", "CONTRADICTORY", "UNVERIFIABLE"],
        },
        "explanation": {"type": "string"},
    },
    "required": ["verdict", "explanation"],
}

# Confidence to assign when Qwen overrides DeBERTa (prompt produces no numeric confidence)
_RATIONAL_VERDICT_CONFIDENCE = {
    "SUPPORTED": 0.80,
    "REFUTED": 0.80,
    "CONTRADICTORY": 0.55,
    "UNVERIFIABLE": 0.0,
}


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


async def _qwen_rational_check(claim_text: str, evidence_items: list[dict]) -> dict | None:
    """
    When DeBERTa is uncertain, ask Qwen to adjudicate using strict blind-judge logic.
    Returns {"verdict": str, "confidence": float, "explanation": str} or None on failure.
    """
    top = evidence_items[:3]
    evidence_lines = []
    for i, item in enumerate(top):
        try:
            domain = item["url"].split("/")[2]
        except (IndexError, KeyError):
            domain = item.get("url", "unknown")
        snippet = (item.get("snippet_text") or "")[:400]
        evidence_lines.append(f"[{i + 1}] Source ({domain}): \"{snippet}\"")

    user_prompt = (
        f"CLAIM: \"{claim_text}\"\n\n"
        f"EVIDENCE:\n" + "\n".join(evidence_lines)
    )

    try:
        data = await call_ollama(
            messages=[
                {"role": "system", "content": _RATIONAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            schema=_RATIONAL_SCHEMA,
        )
        verdict = str(data.get("verdict", "")).upper()
        if verdict not in ("SUPPORTED", "REFUTED", "CONTRADICTORY", "UNVERIFIABLE"):
            return None
        explanation = str(data.get("explanation", "")).strip()
        confidence = _RATIONAL_VERDICT_CONFIDENCE.get(verdict, 0.5)
        return {"verdict": verdict, "confidence": confidence, "explanation": explanation}
    except Exception as exc:
        _logger.warning("Rational NLI check failed: %s", exc)
        return None


async def evaluate_fact(claim: ClaimObject, evidence_items: list[dict], claim_domain: str = "general") -> dict:
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

        weighted_confidence_sum += source_confidence * weight
        weighted_entailment_sum += p_entailment * weight
        weighted_contradiction_sum += p_contradiction * weight
        weighted_neutral_sum += p_neutral * weight
        weight_total += weight

    composite_confidence_score = weighted_confidence_sum / weight_total
    avg_entailment = weighted_entailment_sum / weight_total
    avg_contradiction = weighted_contradiction_sum / weight_total
    avg_neutral = weighted_neutral_sum / weight_total

    if avg_entailment >= 0.60 and avg_entailment > avg_contradiction:
        verdict_label = "SUPPORTED"
    elif avg_contradiction >= 0.60 and avg_contradiction > avg_entailment:
        verdict_label = "REFUTED"
    elif abs(avg_entailment - avg_contradiction) <= 0.15 and avg_entailment > 0.35 and avg_contradiction > 0.35:
        verdict_label = "CONTRADICTORY"
    else:
        verdict_label = "UNVERIFIABLE"

    # The DeBERTa cross-encoder has two known blind spots:
    #   - lower-bound numbers: it refutes "more than 76" when evidence says "500"
    #   - topical single sources: it entails on shared vocabulary, not the proposition
    # So we run Qwen's strict blind-judge as a CONFIRMATION GATE on every affirmative
    # verdict — not only when DeBERTa is uncertain. The gate may only CORRECT a verdict:
    # on agreement it keeps the stronger confidence (never downgrades a good verdict).
    rational_explanation: str | None = None
    rational_threshold = 0.45 if claim_domain in ("political", "current_affairs") else _RATIONAL_THRESHOLD

    has_number = bool(re.search(r"\d", claim.claim_text))
    single_source = len(evidence_items) == 1
    uncertain = verdict_label == "UNVERIFIABLE" or composite_confidence_score < rational_threshold
    confirm_strong = verdict_label in ("SUPPORTED", "REFUTED")

    if uncertain or confirm_strong:
        rational = await _qwen_rational_check(claim.claim_text, evidence_items)
        if rational is not None:
            r_verdict = rational["verdict"]
            if r_verdict == verdict_label:
                # Agreement → keep the stronger confidence; never weaken a verdict the judge backs
                composite_confidence_score = max(composite_confidence_score, rational["confidence"])
                rational_explanation = rational["explanation"] or None
            elif r_verdict != "UNVERIFIABLE":
                # Concrete disagreement → the blind-judge corrects the cross-encoder
                # (e.g. flips a false REFUTED on a "more than N" claim back to SUPPORTED)
                verdict_label = r_verdict
                composite_confidence_score = (
                    0.35 * composite_confidence_score + 0.65 * rational["confidence"]
                )
                rational_explanation = rational["explanation"] or None
            elif confirm_strong and (has_number or single_source):
                # Judge found the evidence doesn't actually address the claim. Demote a strong
                # verdict only when it was RISKY (numeric or single-source) so robust
                # multi-source verdicts are never weakened by a terse-snippet read.
                verdict_label = "UNVERIFIABLE"
                composite_confidence_score = 0.0
                rational_explanation = rational["explanation"] or None

    return {
        "verdict_label": verdict_label,
        "composite_confidence_score": composite_confidence_score,
        "evidence_sources": evidence_sources,
        "avg_entailment": avg_entailment,
        "avg_contradiction": avg_contradiction,
        "avg_neutral": avg_neutral,
        "rational_explanation": rational_explanation,
    }
