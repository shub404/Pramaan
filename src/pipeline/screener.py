import re

_NUMBER_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s*(?:%|percent|million|billion|trillion|thousand|hundred|km|kg|mph|mph|°C|°F))?\b"
)
_PROPER_NOUN_RE = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+")
_FACTUAL_VERB_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|found|shows?|show|reveals?|reveal|"
    r"reported?|report|according|study|studies|research|data|scientists?|"
    r"percent|million|billion|trillion|government|published|announced|"
    r"confirmed|demonstrated|measured|estimated|calculated|increased|"
    r"decreased|caused|linked|associated|discovered|developed|launched)\b",
    re.IGNORECASE,
)

_FILLER_FIRST_WORDS = frozenset(
    "hi hello hey welcome thanks thank okay ok um uh alright anyway".split()
)

_OPINION_PREFIXES = (
    "i think", "i believe", "i feel", "i guess", "i wonder",
    "in my opinion", "personally", "maybe ", "perhaps ",
    "it seems", "it appears", "i'm not sure",
)


def is_verifiable_claim(sentence: str) -> bool:
    """Return True if the sentence is a verifiable factual claim worth checking."""
    text = sentence.strip()
    if not text:
        return False

    words = text.split()
    word_count = len(words)
    if word_count < 5 or word_count > 100:
        return False

    if text.endswith("?"):
        return False

    if words[0].lower() in _FILLER_FIRST_WORDS:
        return False

    lower = text.lower()
    if any(lower.startswith(p) for p in _OPINION_PREFIXES):
        return False

    has_number = bool(_NUMBER_RE.search(text))
    has_proper_noun = bool(_PROPER_NOUN_RE.search(text))
    has_factual_verb = bool(_FACTUAL_VERB_RE.search(text))

    return has_number or has_proper_noun or has_factual_verb
