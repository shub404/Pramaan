from datetime import datetime

from pydantic import BaseModel


class ClaimObject(BaseModel):
    claim_id: str
    timestamp_start: int
    timestamp_end: int
    claim_text: str
    importance_score: int
    verification_status: str


class EvidenceSource(BaseModel):
    url: str
    snippet_text: str
    domain_tier: int
    nli_entailment_prob: float
    nli_contradiction_prob: float


class FinalVerdictPacket(BaseModel):
    video_url: str
    processed_at: datetime
    claim: ClaimObject
    top_evidence_sources: list[EvidenceSource]
    composite_confidence_score: float
    verdict_label: str
    explanation_summary: str