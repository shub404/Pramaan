import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import EMBEDDING_MODEL, NLI_MODEL

_embedding_model: SentenceTransformer | None = None
_nli_tokenizer: AutoTokenizer | None = None
_nli_model: AutoModelForSequenceClassification | None = None


def get_embedding_model() -> SentenceTransformer:   # to retrieve embedding model easily in project
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedding_model


def get_nli_components() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:   # to retrieve nle model easily in project
    global _nli_tokenizer, _nli_model
    if _nli_tokenizer is None or _nli_model is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        _nli_model = _nli_model.to("cpu")
        _nli_model.eval()
    return _nli_tokenizer, _nli_model
