"""Abstraction over LLM API calls (Ollama by default).

Provides a unified interface for making LLM calls with deterministic
settings. Designed to allow swapping backends without changing caller code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""

    backend: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.0
    seed: int = 42
    timeout: int = 120


class LLMClient:
    """Client for making LLM API calls.

    Wraps Ollama's HTTP API with deterministic defaults.
    Backend-agnostic interface so callers don't depend on Ollama specifics.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

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
        if self.config.backend == "ollama":
            return self._generate_ollama(prompt, system, **kwargs)
        raise LLMError(f"Unknown backend: {self.config.backend}")

    def generate_json(self, prompt: str, system: str = "", **kwargs: Any) -> dict:
        """Generate a completion and parse it as JSON.

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
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse LLM response as JSON: {e}") from e

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
        if "format" in kwargs:
            payload["format"] = kwargs["format"]

        logger.debug("Ollama request to %s model=%s", url, self.config.model)

        try:
            resp = requests.post(url, json=payload, timeout=self.config.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Ollama API call failed: {e}") from e

        data = resp.json()
        return data.get("response", "")


class LLMError(Exception):
    """Raised when an LLM API call fails."""


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
