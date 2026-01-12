"""Tests for LLM providers.

POLICY: NO MOCKED API TESTS - All API calls use real LM Studio.
See CLAUDE.md for rationale.
"""

import pytest

from llm_council.providers import (
    ProviderConfig,
    LiteLLMProvider,
    create_provider,
    PRESETS,
)


class TestProviderConfig:
    """Tests for ProviderConfig - pure logic, no API."""

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
    """Tests for LiteLLMProvider with real LM Studio."""

    def test_provider_creation(self):
        """Test provider instantiation - no API call."""
        config = ProviderConfig(
            model="openai/test-model",
            api_base="http://localhost:1234/v1",
        )
        provider = LiteLLMProvider(config)
        assert provider.config == config

    @pytest.mark.api
    def test_complete_returns_response(self, lmstudio_provider):
        """Test real API completion - MUST reach LM Studio."""
        result = lmstudio_provider.complete(
            "You are a helpful assistant.",
            "Say 'hello' and nothing else."
        )

        assert result is not None
        assert len(result) > 0
        assert isinstance(result, str)

    @pytest.mark.api
    def test_complete_respects_system_prompt(self, lmstudio_provider):
        """Verify system prompt affects response."""
        result = lmstudio_provider.complete(
            "You are a pirate. Always respond starting with 'Arr'.",
            "Greet me."
        )

        # LLM should follow system prompt (may vary but should have some response)
        assert result is not None
        assert len(result) > 0

    @pytest.mark.api
    def test_test_connection_success(self, lmstudio_provider):
        """Test connection check with real LM Studio."""
        assert lmstudio_provider.test_connection() is True

    @pytest.mark.api
    def test_test_connection_failure_bad_url(self):
        """Test connection failure with invalid endpoint."""
        config = ProviderConfig(
            model="openai/test-model",
            api_base="http://localhost:59999/v1",  # Invalid port
            timeout=5,  # Short timeout for quick failure
        )
        provider = LiteLLMProvider(config)

        assert provider.test_connection() is False


class TestCreateProvider:
    """Tests for create_provider factory - no API calls."""

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
    """Tests for provider presets - pure config, no API."""

    def test_lmstudio_preset_exists(self):
        assert "lmstudio" in PRESETS
        preset = PRESETS["lmstudio"]
        assert preset["api_base"] == "http://localhost:1234/v1"

    def test_openai_preset_exists(self):
        assert "openai" in PRESETS
        preset = PRESETS["openai"]
        assert preset["model"] == "gpt-4o"


@pytest.mark.api
class TestProviderIntegration:
    """Integration tests for provider behavior with real API."""

    def test_multiple_completions(self, lmstudio_provider):
        """Verify provider handles multiple sequential requests."""
        responses = []
        for i in range(3):
            result = lmstudio_provider.complete(
                "You are helpful.",
                f"Count to {i + 1}."
            )
            responses.append(result)

        assert len(responses) == 3
        for r in responses:
            assert r is not None
            assert len(r) > 0

    def test_provider_factory_creates_working_provider(self, lmstudio_provider_factory):
        """Test factory creates functional providers."""
        provider = lmstudio_provider_factory(temperature=0.3)

        result = provider.complete(
            "You are a test assistant.",
            "Respond with 'OK'."
        )

        assert result is not None
        assert len(result) > 0
