import asyncio
import re

import nltk
from youtube_transcript_api import YouTubeTranscriptApi

nltk.download("punkt_tab", quiet=True)

_VIDEO_ID_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=)([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/(?:embed|shorts|live)/)([a-zA-Z0-9_-]{11})"),
]

_WORD_TARGET = 500
_SENTENCE_OVERLAP = 2


def extract_video_id(url: str) -> str:          # used to extract 11 digit video id from url
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot extract a valid YouTube video ID from: {url}")


def _build_sentences(fragments: list[dict]) -> list[dict]:      # input as list of fragments, and output as complete sentences with start and end timestamp
    text_parts: list[str] = []  
    offsets: list[tuple[int, float, float]] = []
    current_char = 0

    for fragment in fragments:
        text = fragment.get("text", "").strip()
        if not text:
            continue
        frag_start = float(fragment["start"])
        frag_end = frag_start + float(fragment.get("duration", 0))
        offsets.append((current_char, frag_start, frag_end))
        text_parts.append(text)
        current_char += len(text) + 1

    if not offsets:
        return []

    full_text = " ".join(text_parts)                # join each element in text_parts with " "
    raw_sentences = nltk.sent_tokenize(full_text)     #  split full_text into sentences

    sentences: list[dict] = []
    search_from = 0
    # to convert (0, 0, 2), (12, 2, 4) into 
    # {"text": "Hello world", "start": 0, "end": 2}..
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue

        pos = full_text.find(sent, search_from)
        if pos == -1:
            pos = search_from

        start_time = offsets[0][1]
        for char_start, frag_start, _ in offsets:
            if char_start <= pos:
                start_time = frag_start
            else:
                break

        end_pos = pos + len(sent) - 1
        end_time = offsets[-1][2]                   # -1 becuase we need end time of last fragment (which corresponds to the last character of the sentence)
        for char_start, _, frag_end in offsets:     # _ means here we dont care about that value
            if char_start <= end_pos:
                end_time = frag_end
            else:
                break

        sentences.append({"text": sent, "start": start_time, "end": end_time})
        search_from = pos + len(sent)

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


async def get_raw_fragments(video_url: str) -> list[dict]:
    video_id = extract_video_id(video_url)

    try:
        api = YouTubeTranscriptApi()
        fetched = await asyncio.to_thread(api.fetch, video_id)
        return [
            {"text": s.text, "start": s.start, "duration": s.duration}
            for s in fetched
        ]
    except Exception as exc:
        raise ValueError(f"Failed to retrieve transcript for {video_id}: {exc}") from exc
