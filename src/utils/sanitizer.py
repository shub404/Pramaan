import json
import re


def clean_and_parse_json(raw_text: str) -> dict:
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    text = fence_match.group(1) if fence_match else raw_text
    text = text.strip()

    first_open = -1
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            first_open = i
            break

    last_close = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ("}", "]"):
            last_close = i
            break

    if first_open == -1 or last_close == -1 or first_open >= last_close:
        raise ValueError("No valid JSON boundaries found in LLM response.")

    candidate = text[first_open : last_close + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        comma_cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(comma_cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON parsing failed after boundary extraction and trailing-comma removal: {exc}"
            ) from exc
