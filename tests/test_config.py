"""Tests for config loading and LLM client validation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.config import load_config
from src.utils.llm import LLMClient, LLMConfig


# ---------------------------------------------------------------------------
# Config loading tests (Phase 3C)
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_empty_file_raises(self, tmp_path):
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_config(cfg)

    def test_malformed_yaml_raises(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(":\n  - :\n    :\n  [invalid")
        with pytest.raises(ValueError, match="Malformed YAML"):
            load_config(cfg)

    def test_missing_required_keys_raises(self, tmp_path):
        cfg = tmp_path / "partial.yaml"
        cfg.write_text("llm:\n  backend: ollama\n")
        with pytest.raises(ValueError, match="missing required"):
            load_config(cfg)

    def test_valid_config_loads(self, tmp_path):
        cfg = tmp_path / "good.yaml"
        cfg.write_text("generation:\n  batch_size: 5\nvalidation:\n  quality_threshold: 0.8\n")
        result = load_config(cfg)
        assert result["generation"]["batch_size"] == 5

    def test_non_dict_yaml_raises(self, tmp_path):
        cfg = tmp_path / "list.yaml"
        cfg.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(cfg)


# ---------------------------------------------------------------------------
# API key validation tests (Phase 3A)
# ---------------------------------------------------------------------------

class TestAPIKeyValidation:
    def test_openai_without_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure env var is not set
            os.environ.pop("OPENAI_API_KEY", None)
            config = LLMConfig(backend="openai", api_key="")
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                LLMClient(config)

    def test_anthropic_without_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = LLMConfig(backend="anthropic", api_key="")
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                LLMClient(config)

    def test_gemini_without_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            config = LLMConfig(backend="gemini", api_key="")
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                LLMClient(config)

    def test_ollama_unreachable_raises(self):
        config = LLMConfig(backend="ollama", base_url="http://localhost:99999")
        with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
            LLMClient(config)
