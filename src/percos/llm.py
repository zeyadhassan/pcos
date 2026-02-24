"""LLM integration – thin async wrapper around OpenAI-compatible chat completions."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from percos.config import get_settings
from percos.logging import get_logger

log = get_logger("llm")


class LLMClient:
    """Async LLM client with retry logic."""

    def __init__(self) -> None:
        settings = get_settings()
        kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = settings.openai_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant's reply."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        log.debug("llm_response", tokens=response.usage.total_tokens if response.usage else 0)
        return content

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> dict | list:
        """Chat completion that returns parsed JSON."""
        raw = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return json.loads(raw)  # type: ignore[return-value]

    async def extract_structured(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.1,
    ) -> dict | list:
        """Convenience wrapper: system prompt + user content → JSON."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return await self.chat_json(messages, temperature=temperature)


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm
