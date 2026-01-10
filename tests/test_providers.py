"""Tests for LLM providers."""

import pytest
from unittest.mock import patch, MagicMock

from llm_council.providers import (
    ProviderConfig,
    LiteLLMProvider,
    create_provider,
    PRESETS,
)


class TestProviderConfig:
    """Tests for ProviderConfig."""

    def test_config_defaults(self):
        config = ProviderConfig(model="test-model")
        assert config.model == "test-model"
        assert config.api_base is None
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.timeout == 120

    def test_config_custom_values(self):
        config = ProviderConfig(
            model="custom-model",
            api_base="http://localhost:1234/v1",
            api_key="test-key",
            temperature=0.5,
            max_tokens=2048,
        )
        assert config.api_base == "http://localhost:1234/v1"
        assert config.api_key == "test-key"
        assert config.temperature == 0.5


class TestLiteLLMProvider:
    """Tests for LiteLLMProvider."""

    def test_provider_creation(self):
        config = ProviderConfig(
            model="openai/test-model",
            api_base="http://localhost:1234/v1",
        )
        provider = LiteLLMProvider(config)
        assert provider.config == config

    @patch("llm_council.providers.litellm.completion")
    def test_complete_calls_litellm(self, mock_completion):
        # Setup mock
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_completion.return_value = mock_response

        config = ProviderConfig(model="test-model")
        provider = LiteLLMProvider(config)

        result = provider.complete("System prompt", "User prompt")

        assert result == "Test response"
        mock_completion.assert_called_once()

    @patch("llm_council.providers.litellm.completion")
    def test_test_connection_success(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_completion.return_value = mock_response

        config = ProviderConfig(model="test-model")
        provider = LiteLLMProvider(config)

        assert provider.test_connection() is True

    @patch("llm_council.providers.litellm.completion")
    def test_test_connection_failure(self, mock_completion):
        mock_completion.side_effect = Exception("Connection failed")

        config = ProviderConfig(model="test-model")
        provider = LiteLLMProvider(config)

        assert provider.test_connection() is False


class TestCreateProvider:
    """Tests for create_provider factory."""

    def test_create_litellm_provider(self):
        provider = create_provider(
            provider_type="litellm",
            model="test-model",
            api_base="http://localhost:1234/v1",
        )
        assert isinstance(provider, LiteLLMProvider)

    def test_create_provider_default_type(self):
        provider = create_provider(model="test-model")
        assert isinstance(provider, LiteLLMProvider)


class TestPresets:
    """Tests for provider presets."""

    def test_lmstudio_preset_exists(self):
        assert "lmstudio" in PRESETS
        preset = PRESETS["lmstudio"]
        assert preset["api_base"] == "http://localhost:1234/v1"

    def test_openai_preset_exists(self):
        assert "openai" in PRESETS
        preset = PRESETS["openai"]
        assert preset["model"] == "gpt-4o"
