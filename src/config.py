import os
from pathlib import Path

# these two paths are set at beginning, so that when we load any ML lib (transformer, torch),
# they know where to download the models as they only check once while being imported

os.environ["HF_HOME"] = "D:\\Pramaan_Storage\\hf_cache"
os.environ["OLLAMA_MODELS"] = "D:\\Pramaan_Storage\\ollama_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

BASE_DIR: Path = Path(__file__).parent.parent.resolve()
DB_PATH: str = str(BASE_DIR / "pramaan.db")

OLLAMA_BASE_URL: str = "http://localhost:11434"
LLM_MODEL: str = "qwen2.5:3b"                           # ollama model
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"    # NLI model for verification
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"               # embedding model to convert text into numbers and check similarity

MAX_CONCURRENT_VERIFICATIONS: int = 3                   # max claims being verified in parallel
CLAIM_EXTRACTION_STEP: int = 15                         # seconds of new transcript before triggering Qwen extraction
CLAIM_EXTRACTION_OVERLAP: int = 5                       # seconds of look-back overlap between extraction windows
DDG_MAX_RESULTS: int = 5                                # DDG search results per claim
EVIDENCE_TOP_K: int = 3                                 # top-K snippets kept after cosine filter
COSINE_SIMILARITY_THRESHOLD: float = 0.65               # to check similarity between two embedding/claims, above .65 means similar
SEMANTIC_DEDUPLICATION_THRESHOLD: float = 0.85          # if two claims have >0.85 cosine, then both are considered 1 (avoids duplications)
JOB_TIMEOUT_SECONDS: int = 120                          # timeout for a job

# more weights assigned to trusted domains 
DOMAIN_TIER_WEIGHTS: dict[str, float] = {
    "gov": 1.0,
    "edu": 1.0,
    "int": 1.0,
    "wikipedia.org": 0.85,
    "reuters.com": 0.85,
    "bbc.com": 0.85,
}
