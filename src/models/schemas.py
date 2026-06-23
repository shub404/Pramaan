from datetime import datetime

from pydantic import BaseModel                  # used to define base class to define and check schema/structures


class ClaimObject(BaseModel):                   # for the Youtube video 
    claim_id: str
    timestamp_start: int
    timestamp_end: int
    claim_text: str
    importance_score: int
    verification_status: str


class EvidenceSource(BaseModel):                # for the evidence website returned by search engine
    url: str
    snippet_text: str
    domain_tier: int                            # how trusted is the website
    nli_entailment_prob: float                  # probability returned by NLI of how similar are both Claim & Evidence
    nli_contradiction_prob: float               # probability of how contradictory are both Claim & Evidence


class FinalVerdictPacket(BaseModel):
    video_url: str
    processed_at: datetime
    claim: ClaimObject                          # object of ClaimObject class
    top_evidence_sources: list[EvidenceSource]  # list of EvidenceSource class as 1 claim has many evidence sites
    composite_confidence_score: float           # final weighted score
    verdict_label: str
    explanation_summary: str