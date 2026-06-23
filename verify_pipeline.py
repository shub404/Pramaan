import asyncio
import traceback

from src.pipeline.transcriber import get_overlapping_chunks
from src.pipeline.extractor import deduplicate_claims, extract_claims_from_chunks
from src.pipeline.retriever import get_relevant_evidence
from src.pipeline.analyzer import evaluate_fact

TEST_URL = "https://www.youtube.com/watch?v=sNhhvQGsMEc"

_DIVIDER = "=" * 64


async def main():
    print(_DIVIDER)
    print("  PRAMAAN — END-TO-END PIPELINE DIAGNOSTIC")
    print(_DIVIDER)
    print(f"  Target : {TEST_URL}")
    print(_DIVIDER)

    print("\n[STAGE 1] Transcript Ingestion & Sliding-Window Slicing")
    chunks = await get_overlapping_chunks(TEST_URL)
    print(f"  Windows constructed : {len(chunks)}")

    print("\n[STAGE 2] Claim Extraction via Qwen 2.5 3B")
    claims = await extract_claims_from_chunks(chunks, TEST_URL)
    print(f"  Claims extracted    : {len(claims)}")

    print("\n[STAGE 3] Semantic Deduplication & Priority Ranking")
    deduplicated = deduplicate_claims(claims)
    print(f"  Claims after dedup  : {len(deduplicated)}")
    print()
    for idx, claim in enumerate(deduplicated, start=1):
        print(f"  [{idx}] importance={claim.importance_score}  |  {claim.claim_text}")

    print(f"\n{_DIVIDER}")
    print("[STAGE 4] Retrieval, NLI Verification & Verdict Resolution")
    print(_DIVIDER)

    for idx, claim in enumerate(deduplicated, start=1):
        print(f"\n  Claim [{idx}]")
        print(f"  Text : {claim.claim_text}")

        evidence = await get_relevant_evidence(claim.claim_text)
        print(f"  Evidence sentences cleared threshold : {len(evidence)}")

        result = await evaluate_fact(claim, evidence)

        tiers = [src.domain_tier for src in result["evidence_sources"]]

        print(f"  Entailment           : {result['avg_entailment']:.4f}")
        print(f"  Contradiction        : {result['avg_contradiction']:.4f}")
        print(f"  Neutral              : {result['avg_neutral']:.4f}")
        print(f"  Domain Tiers         : {tiers if tiers else 'N/A — no evidence retrieved'}")
        print(f"  Composite Confidence : {result['composite_confidence_score']:.4f}")
        print(f"  Verdict              : {result['verdict_label']}")

    print(f"\n{_DIVIDER}")
    print("  DIAGNOSTIC COMPLETE")
    print(_DIVIDER)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        print("\n[FATAL] Pipeline raised an unhandled exception:")
        print(traceback.format_exc())
