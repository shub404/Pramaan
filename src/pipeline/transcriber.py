import asyncio
import re

from youtube_transcript_api import YouTubeTranscriptApi

_VIDEO_ID_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=)([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/(?:embed|shorts|live)/)([a-zA-Z0-9_-]{11})"),
]

_SENTENCE_BOUNDARY = re.compile(r"[.?!]\s*$")
_WORD_TARGET = 500
_SENTENCE_OVERLAP = 2


def extract_video_id(url: str) -> str:
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot extract a valid YouTube video ID from: {url}")


def _build_sentences(fragments: list[dict]) -> list[dict]:
    sentences = []
    buffer_text = ""
    buffer_start: float | None = None
    buffer_end = 0.0

    for fragment in fragments:
        raw_text = fragment.get("text", "").strip()
        if not raw_text:
            continue

        start = float(fragment["start"])
        duration = float(fragment.get("duration", 0))

        if buffer_start is None:
            buffer_start = start

        buffer_text = (buffer_text + " " + raw_text).strip()
        buffer_end = start + duration

        if _SENTENCE_BOUNDARY.search(buffer_text):
            sentences.append({
                "text": buffer_text,
                "start": buffer_start,
                "end": buffer_end,
            })
            buffer_text = ""
            buffer_start = None

    if buffer_text and buffer_start is not None:
        sentences.append({
            "text": buffer_text,
            "start": buffer_start,
            "end": buffer_end,
        })

    return sentences


def _build_windows(sentences: list[dict]) -> list[dict]:
    windows = []
    i = 0

    while i < len(sentences):
        window: list[dict] = []
        word_count = 0

        for sentence in sentences[i:]:
            window.append(sentence)
            word_count += len(sentence["text"].split())
            if word_count >= _WORD_TARGET:
                break

        if not window:
            break

        windows.append({
            "text": " ".join(s["text"] for s in window),
            "timestamp_start": int(window[0]["start"]),
            "timestamp_end": int(window[-1]["end"]),
        })

        i += max(1, len(window) - _SENTENCE_OVERLAP)

    return windows


async def get_overlapping_chunks(video_url: str) -> list[dict]:
    video_id = extract_video_id(video_url)

    try:
        api = YouTubeTranscriptApi()
        fetched = await asyncio.to_thread(api.fetch, video_id)
        fragments = [
            {"text": s.text, "start": s.start, "duration": s.duration}
            for s in fetched
        ]
    except Exception as exc:
        raise ValueError(f"Failed to retrieve transcript for {video_id}: {exc}") from exc

    if not fragments:
        return []

    sentences = _build_sentences(fragments)
    if not sentences:
        return []

    return _build_windows(sentences)
