import asyncio
import json

import httpx

from src.config import LLM_MODEL, OLLAMA_BASE_URL

_ollama_sem = asyncio.Semaphore(1)


async def call_ollama(messages: list[dict], schema: dict, timeout: float = 35.0) -> dict:
    async with _ollama_sem:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": LLM_MODEL, "messages": messages, "format": schema, "stream": False},
            )
            resp.raise_for_status()
            return json.loads(resp.json()["message"]["content"])
