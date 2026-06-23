import os
from pathlib import Path

os.environ["HF_HOME"] = "D:\\Pramaan_Storage\\hf_cache"
os.environ["OLLAMA_MODELS"] = "D:\\Pramaan_Storage\\ollama_cache"

BASE_DIR: Path = Path(__file__).parent.parent.resolve()
DB_PATH: str = str(BASE_DIR / "pramaan.db")

OLLAMA_BASE_URL: str = "http://localhost:11434"
LLM_MODEL: str = "qwen2.5:3b"
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

MAX_CLAIMS_PER_VIDEO: int = 4
COSINE_SIMILARITY_THRESHOLD: float = 0.65
SEMANTIC_DEDUPLICATION_THRESHOLD: float = 0.85
JOB_TIMEOUT_SECONDS: int = 120

DOMAIN_TIER_WEIGHTS: dict[str, float] = {
    "gov": 1.0,
    "edu": 1.0,
    "int": 1.0,
    "wikipedia.org": 0.85,
    "reuters.com": 0.85,
    "bbc.com": 0.85,
}
