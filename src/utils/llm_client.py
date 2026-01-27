"""
LLM Client - OpenAI-compatible API wrapper.

Supports multiple providers via the OpenAI protocol:
- z.AI (primary)
- OpenRouter
- OpenAI
- Any OpenAI-compatible endpoint (Ollama, vLLM, etc.)

All providers are external - not self-hosted.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import httpx


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    ZAI = "zai"              # z.AI / Google AI
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    CUSTOM = "custom"        # Any OpenAI-compatible endpoint


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: LLMProvider
    api_key: str
    base_url: str
    model: str
    timeout: float = 60.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
        provider_str = os.getenv("LLM_PROVIDER", "zai").lower()
        provider = LLMProvider(provider_str)

        # Provider-specific defaults
        defaults = {
            LLMProvider.ZAI: {
                "base_url": "https://api.zai.dev/v1",
                "model": "gemini-2.5-flash",
                "api_key_env": "ZAI_API_KEY",
            },
            LLMProvider.OPENROUTER: {
                "base_url": "https://openrouter.ai/api/v1",
                "model": "google/gemini-flash-1.5",
                "api_key_env": "OPENROUTER_API_KEY",
            },
            LLMProvider.OPENAI: {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "api_key_env": "OPENAI_API_KEY",
            },
            LLMProvider.CUSTOM: {
                "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
                "model": os.getenv("LLM_MODEL", "llama3"),
                "api_key_env": "LLM_API_KEY",
            },
        }

        config = defaults[provider]

        return cls(
            provider=provider,
            api_key=os.getenv(config["api_key_env"], os.getenv("LLM_API_KEY", "")),
            base_url=os.getenv("LLM_BASE_URL", config["base_url"]),
            model=os.getenv("LLM_MODEL", config["model"]),
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        )


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    provider: LLMProvider
    usage: dict
    finish_reason: str


class LLMClient:
    """
    OpenAI-compatible LLM client.

    Works with any provider that supports the OpenAI chat completions API.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM client."""
        self.config = config or LLMConfig.from_env()
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=self._build_headers(),
            timeout=self.config.timeout,
        )

    def _build_headers(self) -> dict:
        """Build request headers based on provider."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        # Provider-specific headers
        if self.config.provider == LLMProvider.OPENROUTER:
            headers["HTTP-Referer"] = "https://apex-rag.local"
            headers["X-Title"] = "Apex RAG System"

        return headers

    async def chat(
        self,
        messages: list[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to config model)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            json_mode: If True, request JSON output

        Returns:
            LLMResponse with the completion
        """
        payload = {
            "model": model or self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = await self._request_with_retry("/chat/completions", payload)

        choice = response["choices"][0]
        return LLMResponse(
            content=choice["message"]["content"],
            model=response.get("model", self.config.model),
            provider=self.config.provider,
            usage=response.get("usage", {}),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Simple completion helper.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments passed to chat()

        Returns:
            The completion text
        """
        messages = []

        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))

        messages.append(ChatMessage(role="user", content=prompt))

        response = await self.chat(messages, **kwargs)
        return response.content

    async def filter_relevance(
        self,
        query: str,
        chunks: list[str],
        threshold: float = 0.5,
    ) -> list[tuple[int, str, float]]:
        """
        Filter chunks by relevance to query.

        This is the primary use case for LLM in Apex RAG - filtering
        retrieved chunks for relevance before sending to the agent.

        Args:
            query: The user's query
            chunks: List of text chunks to evaluate
            threshold: Minimum relevance score (0-1)

        Returns:
            List of (index, chunk, score) for relevant chunks
        """
        if not chunks:
            return []

        system_prompt = """You are a relevance filter. For each chunk, rate its relevance to the query on a scale of 0.0 to 1.0.

Respond with a JSON array of objects: [{"index": 0, "score": 0.8}, ...]

Only include chunks with score >= the threshold.
Be strict - only truly relevant chunks should score high."""

        prompt = f"""Query: {query}

Threshold: {threshold}

Chunks:
"""
        for i, chunk in enumerate(chunks):
            # Truncate long chunks
            truncated = chunk[:500] + "..." if len(chunk) > 500 else chunk
            prompt += f"\n[{i}] {truncated}\n"

        try:
            response = await self.complete(
                prompt,
                system_prompt=system_prompt,
                json_mode=True,
                temperature=0.0,
            )

            import json
            results = json.loads(response)

            return [
                (r["index"], chunks[r["index"]], r["score"])
                for r in results
                if r["score"] >= threshold and r["index"] < len(chunks)
            ]
        except Exception:
            # On error, return all chunks as potentially relevant
            return [(i, chunk, 0.5) for i, chunk in enumerate(chunks)]

    async def _request_with_retry(self, endpoint: str, payload: dict) -> dict:
        """Make request with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in (429, 500, 502, 503, 504):
                    # Retry on rate limit or server errors
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except httpx.RequestError as e:
                last_error = e
                import asyncio
                await asyncio.sleep(2 ** attempt)
                continue

        raise last_error or Exception("Request failed after retries")

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


# Convenience function for one-off completions
async def quick_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    **kwargs,
) -> str:
    """Quick one-off completion without managing client lifecycle."""
    async with LLMClient() as client:
        return await client.complete(prompt, system_prompt, **kwargs)
