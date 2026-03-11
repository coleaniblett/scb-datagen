"""Abstraction over LLM API calls (Ollama, OpenAI, Anthropic, Gemini).

Provides a unified interface for making LLM calls with deterministic
settings. All backends use raw HTTP via `requests` — no SDK dependencies.
API keys are read from config with environment-variable fallback.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Default base URLs per backend
DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}

# Default model names per backend
DEFAULT_MODELS: dict[str, str] = {
    "ollama": "llama3.1:8b",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250929",
    "gemini": "gemini-2.0-flash",
}

# Environment variable names for API keys
API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

# Backends that require an API key
_KEYED_BACKENDS = frozenset(API_KEY_ENV_VARS)


@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""

    backend: str = "ollama"
    base_url: str = ""
    model: str = ""
    temperature: float = 0.0
    seed: int = 42
    timeout: int = 120
    api_key: str = ""
    max_tokens: int = 4096
    max_retries: int = 5
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0

    def __post_init__(self) -> None:
        # Apply backend-specific defaults if not explicitly set
        if not self.base_url:
            self.base_url = DEFAULT_BASE_URLS.get(self.backend, "")
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.backend, "")
        # Resolve API key from env var if not in config
        if not self.api_key and self.backend in API_KEY_ENV_VARS:
            self.api_key = os.environ.get(API_KEY_ENV_VARS[self.backend], "")


class LLMClient:
    """Client for making LLM API calls.

    Supports multiple backends (Ollama, OpenAI, Anthropic, Gemini) through
    a unified interface. All backends use raw HTTP via requests.
    """

    BACKENDS = ("ollama", "openai", "anthropic", "gemini")

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        if self.config.backend not in self.BACKENDS:
            raise LLMError(
                f"Unknown backend: {self.config.backend!r}. "
                f"Supported: {', '.join(self.BACKENDS)}"
            )
        self.validate_connection()

    def validate_connection(self) -> None:
        """Validate that the backend is reachable and configured.

        For keyed backends (OpenAI, Anthropic, Gemini), checks that the
        API key is set and non-empty. For Ollama, makes a lightweight
        request to confirm the server is reachable.

        Raises:
            ValueError: If an API key is required but missing.
            ConnectionError: If Ollama is not reachable.
        """
        backend = self.config.backend

        if backend in _KEYED_BACKENDS:
            if not self.config.api_key:
                env_var = API_KEY_ENV_VARS[backend]
                raise ValueError(
                    f"{env_var} environment variable is not set and no api_key "
                    f"was provided in config. Set {env_var} or add api_key to "
                    f"the llm config section."
                )

        if backend == "ollama":
            try:
                resp = requests.get(
                    f"{self.config.base_url}/api/tags",
                    timeout=(5, 10),
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                raise ConnectionError(
                    f"Cannot reach Ollama at {self.config.base_url}. "
                    f"Is the Ollama server running? Error: {e}"
                ) from e

    def generate(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        """Generate a completion from the LLM.

        Args:
            prompt: The user/input prompt.
            system: Optional system prompt.
            **kwargs: Override any generation parameter (temperature, seed, etc.).

        Returns:
            The generated text response.

        Raises:
            LLMError: If the API call fails.
        """
        dispatch = {
            "ollama": self._generate_ollama,
            "openai": self._generate_openai,
            "anthropic": self._generate_anthropic,
            "gemini": self._generate_gemini,
        }
        logger.debug("[LLM-CALL] backend=%s model=%s prompt_len=%d system_len=%d",
                      self.config.backend, self.config.model, len(prompt), len(system))
        logger.debug("[LLM-CALL] prompt:\n%s", prompt)
        if system:
            logger.debug("[LLM-CALL] system:\n%s", system)

        result = dispatch[self.config.backend](prompt, system, **kwargs)

        logger.debug("[LLM-RESPONSE] len=%d:\n%s", len(result), result)
        return result

    def generate_json(self, prompt: str, system: str = "", **kwargs: Any) -> dict:
        """Generate a completion and parse it as JSON.

        Requests JSON format from backends that support it natively.
        Falls back to extracting JSON from freeform text for others.

        Args:
            prompt: The user/input prompt.
            system: Optional system prompt.
            **kwargs: Override any generation parameter.

        Returns:
            Parsed JSON dict from the response.

        Raises:
            LLMError: If the API call or JSON parsing fails.
        """
        raw = self.generate(prompt, system, format="json", **kwargs)
        return _extract_json(raw)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _generate_ollama(self, prompt: str, system: str, **kwargs: Any) -> str:
        """Call the Ollama /api/generate endpoint."""
        url = f"{self.config.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "seed": kwargs.get("seed", self.config.seed),
            },
        }
        if system:
            payload["system"] = system
        if kwargs.get("format") == "json":
            payload["format"] = "json"

        data = self._post(url, payload)
        return data.get("response", "")

    def _generate_openai(self, prompt: str, system: str, **kwargs: Any) -> str:
        """Call the OpenAI chat completions endpoint.

        Also works with any OpenAI-compatible API (Together, Groq, etc.)
        by changing base_url in config.
        """
        url = f"{self.config.base_url}/chat/completions"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "seed": kwargs.get("seed", self.config.seed),
        }
        if kwargs.get("format") == "json":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        data = self._post(url, payload, headers=headers)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected OpenAI response structure: {e}") from e

    def _generate_anthropic(self, prompt: str, system: str, **kwargs: Any) -> str:
        """Call the Anthropic messages endpoint."""
        url = f"{self.config.base_url}/v1/messages"
        messages = [{"role": "user", "content": prompt}]

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if system:
            payload["system"] = system

        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        data = self._post(url, payload, headers=headers)
        try:
            content_blocks = data["content"]
            return "".join(block["text"] for block in content_blocks if block["type"] == "text")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected Anthropic response structure: {e}") from e

    def _generate_gemini(self, prompt: str, system: str, **kwargs: Any) -> str:
        """Call the Gemini generateContent endpoint."""
        model = self.config.model
        url = f"{self.config.base_url}/v1beta/models/{model}:generateContent"

        contents: list[dict[str, Any]] = []
        if system:
            contents.append({"role": "user", "parts": [{"text": system}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        generation_config: dict[str, Any] = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if kwargs.get("format") == "json":
            generation_config["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        headers = {"Content-Type": "application/json"}
        params = {"key": self.config.api_key}

        data = self._post(url, payload, headers=headers, params=params)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected Gemini response structure: {e}") from e

    # ------------------------------------------------------------------
    # HTTP helper with retry
    # ------------------------------------------------------------------

    def _post(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP POST request with exponential backoff retry.

        Retries on HTTP 429 (rate limit) and 5xx (server error) responses.
        Uses exponential backoff with jitter.
        """
        max_retries = self.config.max_retries
        base_delay = self.config.retry_base_delay
        max_delay = self.config.retry_max_delay
        timeout = (min(self.config.timeout, 30), self.config.timeout)

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    url, json=payload, headers=headers, params=params,
                    timeout=timeout,
                )
                # Retry on rate-limit or server errors
                if resp.status_code == 429 or resp.status_code >= 500:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.5)
                    wait = delay + jitter
                    logger.warning(
                        "HTTP %d from %s (attempt %d/%d), retrying in %.1fs",
                        resp.status_code, self.config.backend, attempt + 1, max_retries + 1, wait,
                    )
                    if attempt < max_retries:
                        time.sleep(wait)
                        continue
                    # Final attempt — fall through to raise
                resp.raise_for_status()
                return resp.json()
            except requests.ConnectionError as e:
                last_error = e
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.5)
                    wait = delay + jitter
                    logger.warning(
                        "Connection error for %s (attempt %d/%d), retrying in %.1fs: %s",
                        self.config.backend, attempt + 1, max_retries + 1, wait, e,
                    )
                    time.sleep(wait)
                    continue
            except requests.RequestException as e:
                last_error = e

        raise LLMError(f"{self.config.backend} API call failed after {max_retries + 1} attempts: {last_error}") from last_error


class LLMError(Exception):
    """Raised when an LLM API call fails."""


def _extract_json(text: str) -> dict:
    """Extract and parse JSON from an LLM response.

    Handles cases where the model wraps JSON in markdown fences
    or adds surrounding commentary.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed JSON dict.

    Raises:
        LLMError: If no valid JSON can be extracted.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise LLMError(f"Failed to extract JSON from LLM response: {text[:200]}...")


def load_llm_from_config(config: dict[str, Any]) -> LLMClient:
    """Create an LLMClient from a config dict (e.g. parsed from defaults.yaml).

    Args:
        config: Dict with keys matching LLMConfig fields, typically
                the 'llm' section of defaults.yaml.

    Returns:
        Configured LLMClient instance.
    """
    llm_config = LLMConfig(**{k: v for k, v in config.items() if k in LLMConfig.__dataclass_fields__})
    return LLMClient(llm_config)
