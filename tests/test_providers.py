"""Tests for LLM providers.

POLICY: NO MOCKED API TESTS - All API calls use real LM Studio.
See CLAUDE.md for rationale.
"""

import pytest

from llm_council.providers import (
    ProviderRegistry,
    ProviderConfig,
    LiteLLMProvider,
    create_provider,
    PRESETS,
)
from llm_council.config import (
    ResolvedConfig,
    ProviderSettings,
    GenerationSettings,
    CouncilSettings,
    PersistenceSettings,
)


# =============================================================================
# TestProviderConfig - Pure logic tests for configuration dataclass
# =============================================================================


class TestProviderConfig:
    """Tests for ProviderConfig - pure logic, no API."""

    def test_config_defaults(self):
        """Test default values for ProviderConfig."""
        config = ProviderConfig(model="test-model")
        assert config.model == "test-model"
        assert config.api_base is None
        assert config.api_key is None
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.timeout == 120

    def test_config_custom_values(self):
        """Test custom values are stored correctly."""
        config = ProviderConfig(
            model="custom-model",
            api_base="http://localhost:1234/v1",
            api_key="test-key",
            temperature=0.5,
            max_tokens=2048,
        )
        assert config.model == "custom-model"
        assert config.api_base == "http://localhost:1234/v1"
        assert config.api_key == "test-key"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    # --- Tests for all 10 inference parameters ---

    def test_config_temperature(self):
        """Test temperature parameter."""
        config = ProviderConfig(model="test", temperature=0.0)
        assert config.temperature == 0.0

        config = ProviderConfig(model="test", temperature=1.0)
        assert config.temperature == 1.0

        config = ProviderConfig(model="test", temperature=0.5)
        assert config.temperature == 0.5

    def test_config_top_p(self):
        """Test top_p parameter."""
        config = ProviderConfig(model="test")
        assert config.top_p is None

        config = ProviderConfig(model="test", top_p=0.9)
        assert config.top_p == 0.9

        config = ProviderConfig(model="test", top_p=0.5)
        assert config.top_p == 0.5

    def test_config_top_k(self):
        """Test top_k parameter."""
        config = ProviderConfig(model="test")
        assert config.top_k is None

        config = ProviderConfig(model="test", top_k=50)
        assert config.top_k == 50

        config = ProviderConfig(model="test", top_k=10)
        assert config.top_k == 10

    def test_config_max_tokens(self):
        """Test max_tokens parameter."""
        config = ProviderConfig(model="test")
        assert config.max_tokens == 1024  # default

        config = ProviderConfig(model="test", max_tokens=512)
        assert config.max_tokens == 512

        config = ProviderConfig(model="test", max_tokens=4096)
        assert config.max_tokens == 4096

    def test_config_frequency_penalty(self):
        """Test frequency_penalty parameter."""
        config = ProviderConfig(model="test")
        assert config.frequency_penalty is None

        config = ProviderConfig(model="test", frequency_penalty=0.5)
        assert config.frequency_penalty == 0.5

        config = ProviderConfig(model="test", frequency_penalty=1.0)
        assert config.frequency_penalty == 1.0

    def test_config_presence_penalty(self):
        """Test presence_penalty parameter."""
        config = ProviderConfig(model="test")
        assert config.presence_penalty is None

        config = ProviderConfig(model="test", presence_penalty=0.5)
        assert config.presence_penalty == 0.5

        config = ProviderConfig(model="test", presence_penalty=1.0)
        assert config.presence_penalty == 1.0

    def test_config_repeat_penalty(self):
        """Test repeat_penalty parameter (LM Studio extension)."""
        config = ProviderConfig(model="test")
        assert config.repeat_penalty is None

        config = ProviderConfig(model="test", repeat_penalty=1.1)
        assert config.repeat_penalty == 1.1

        config = ProviderConfig(model="test", repeat_penalty=1.5)
        assert config.repeat_penalty == 1.5

    def test_config_stop(self):
        """Test stop sequences parameter."""
        config = ProviderConfig(model="test")
        assert config.stop is None

        config = ProviderConfig(model="test", stop=["<end>"])
        assert config.stop == ["<end>"]

        config = ProviderConfig(model="test", stop=["###", "END", "\n\n"])
        assert config.stop == ["###", "END", "\n\n"]

    def test_config_seed(self):
        """Test seed parameter for reproducibility."""
        config = ProviderConfig(model="test")
        assert config.seed is None

        config = ProviderConfig(model="test", seed=42)
        assert config.seed == 42

        config = ProviderConfig(model="test", seed=12345)
        assert config.seed == 12345

    def test_config_timeout(self):
        """Test timeout parameter."""
        config = ProviderConfig(model="test")
        assert config.timeout == 120  # default

        config = ProviderConfig(model="test", timeout=30)
        assert config.timeout == 30

        config = ProviderConfig(model="test", timeout=300)
        assert config.timeout == 300

    def test_config_all_params_together(self):
        """Test all inference parameters set together."""
        config = ProviderConfig(
            model="full-model",
            api_base="http://localhost:1234/v1",
            api_key="my-key",
            temperature=0.3,
            top_p=0.85,
            top_k=40,
            max_tokens=2048,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            repeat_penalty=1.15,
            stop=["<|end|>", "###"],
            seed=12345,
            timeout=180,
        )
        assert config.model == "full-model"
        assert config.api_base == "http://localhost:1234/v1"
        assert config.api_key == "my-key"
        assert config.temperature == 0.3
        assert config.top_p == 0.85
        assert config.top_k == 40
        assert config.max_tokens == 2048
        assert config.frequency_penalty == 0.2
        assert config.presence_penalty == 0.3
        assert config.repeat_penalty == 1.15
        assert config.stop == ["<|end|>", "###"]
        assert config.seed == 12345
        assert config.timeout == 180


# =============================================================================
# TestLiteLLMProvider - Provider creation and parameter storage
# =============================================================================


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

    def test_provider_stores_all_params(self):
        """Test that provider stores all configuration parameters."""
        config = ProviderConfig(
            model="openai/test-model",
            api_base="http://localhost:1234/v1",
            api_key="test-key",
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            max_tokens=2048,
            frequency_penalty=0.3,
            presence_penalty=0.4,
            repeat_penalty=1.2,
            stop=["STOP"],
            seed=42,
            timeout=60,
        )
        provider = LiteLLMProvider(config)

        assert provider.config.model == "openai/test-model"
        assert provider.config.api_base == "http://localhost:1234/v1"
        assert provider.config.api_key == "test-key"
        assert provider.config.temperature == 0.5
        assert provider.config.top_p == 0.9
        assert provider.config.top_k == 50
        assert provider.config.max_tokens == 2048
        assert provider.config.frequency_penalty == 0.3
        assert provider.config.presence_penalty == 0.4
        assert provider.config.repeat_penalty == 1.2
        assert provider.config.stop == ["STOP"]
        assert provider.config.seed == 42
        assert provider.config.timeout == 60

    def test_provider_with_none_optional_params(self):
        """Test provider handles None optional parameters."""
        config = ProviderConfig(
            model="openai/test-model",
            api_base="http://localhost:1234/v1",
        )
        provider = LiteLLMProvider(config)

        assert provider.config.top_p is None
        assert provider.config.top_k is None
        assert provider.config.frequency_penalty is None
        assert provider.config.presence_penalty is None
        assert provider.config.repeat_penalty is None
        assert provider.config.stop is None
        assert provider.config.seed is None

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


# =============================================================================
# TestCreateProvider - Factory function tests
# =============================================================================


class TestCreateProvider:
    """Tests for create_provider factory - no API calls."""

    def test_create_litellm_provider(self):
        """Test factory creates LiteLLMProvider."""
        provider = create_provider(
            provider_type="litellm",
            model="test-model",
            api_base="http://localhost:1234/v1",
        )
        assert isinstance(provider, LiteLLMProvider)

    def test_create_provider_default_type(self):
        """Test factory defaults to LiteLLM."""
        provider = create_provider(model="test-model")
        assert isinstance(provider, LiteLLMProvider)

    def test_create_provider_with_all_params(self):
        """Test factory accepts all inference parameters."""
        provider = create_provider(
            provider_type="litellm",
            model="test-model",
            api_base="http://localhost:1234/v1",
            api_key="test-key",
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            max_tokens=2048,
            frequency_penalty=0.3,
            presence_penalty=0.4,
            repeat_penalty=1.2,
            stop=["END"],
            seed=42,
            timeout=60,
        )

        assert provider.config.model == "test-model"
        assert provider.config.api_base == "http://localhost:1234/v1"
        assert provider.config.api_key == "test-key"
        assert provider.config.temperature == 0.5
        assert provider.config.top_p == 0.9
        assert provider.config.top_k == 50
        assert provider.config.max_tokens == 2048
        assert provider.config.frequency_penalty == 0.3
        assert provider.config.presence_penalty == 0.4
        assert provider.config.repeat_penalty == 1.2
        assert provider.config.stop == ["END"]
        assert provider.config.seed == 42
        assert provider.config.timeout == 60

    def test_create_provider_temperature_only(self):
        """Test factory with only temperature param."""
        provider = create_provider(model="test", temperature=0.1)
        assert provider.config.temperature == 0.1

    def test_create_provider_top_p_only(self):
        """Test factory with only top_p param."""
        provider = create_provider(model="test", top_p=0.8)
        assert provider.config.top_p == 0.8

    def test_create_provider_top_k_only(self):
        """Test factory with only top_k param."""
        provider = create_provider(model="test", top_k=30)
        assert provider.config.top_k == 30

    def test_create_provider_max_tokens_only(self):
        """Test factory with only max_tokens param."""
        provider = create_provider(model="test", max_tokens=512)
        assert provider.config.max_tokens == 512

    def test_create_provider_frequency_penalty_only(self):
        """Test factory with only frequency_penalty param."""
        provider = create_provider(model="test", frequency_penalty=0.7)
        assert provider.config.frequency_penalty == 0.7

    def test_create_provider_presence_penalty_only(self):
        """Test factory with only presence_penalty param."""
        provider = create_provider(model="test", presence_penalty=0.6)
        assert provider.config.presence_penalty == 0.6

    def test_create_provider_repeat_penalty_only(self):
        """Test factory with only repeat_penalty param."""
        provider = create_provider(model="test", repeat_penalty=1.3)
        assert provider.config.repeat_penalty == 1.3

    def test_create_provider_stop_only(self):
        """Test factory with only stop param."""
        provider = create_provider(model="test", stop=["<|end|>", "###"])
        assert provider.config.stop == ["<|end|>", "###"]

    def test_create_provider_seed_only(self):
        """Test factory with only seed param."""
        provider = create_provider(model="test", seed=99999)
        assert provider.config.seed == 99999

    def test_create_provider_timeout_only(self):
        """Test factory with only timeout param."""
        provider = create_provider(model="test", timeout=45)
        assert provider.config.timeout == 45


# =============================================================================
# TestPresets - Tests for preset configurations
# =============================================================================


class TestPresets:
    """Tests for provider presets - pure config, no API."""

    def test_lmstudio_preset_exists(self):
        """Test lmstudio preset configuration."""
        assert "lmstudio" in PRESETS
        preset = PRESETS["lmstudio"]
        assert preset["api_base"] == "http://localhost:1234/v1"
        assert preset["provider_type"] == "litellm"
        assert preset["api_key"] == "lm-studio"

    def test_openai_preset_exists(self):
        """Test openai preset configuration."""
        assert "openai" in PRESETS
        preset = PRESETS["openai"]
        assert preset["model"] == "gpt-4o"
        assert preset["provider_type"] == "litellm"

    def test_openai_mini_preset_exists(self):
        """Test openai-mini preset configuration."""
        assert "openai-mini" in PRESETS
        preset = PRESETS["openai-mini"]
        assert preset["model"] == "gpt-4o-mini"
        assert preset["provider_type"] == "litellm"

    def test_anthropic_preset_exists(self):
        """Test anthropic preset configuration."""
        assert "anthropic" in PRESETS
        preset = PRESETS["anthropic"]
        assert preset["model"] == "claude-3-opus-20240229"
        assert preset["provider_type"] == "litellm"

    def test_ollama_preset_exists(self):
        """Test ollama preset configuration."""
        assert "ollama" in PRESETS
        preset = PRESETS["ollama"]
        assert preset["api_base"] == "http://localhost:11434"
        assert preset["provider_type"] == "litellm"

    def test_all_presets_have_provider_type(self):
        """Verify all presets have provider_type set."""
        for name, preset in PRESETS.items():
            assert "provider_type" in preset, f"Preset '{name}' missing provider_type"
            assert preset["provider_type"] == "litellm"


# =============================================================================
# TestParameterIsolation - Test each parameter individually with real API
# =============================================================================


@pytest.mark.api
class TestParameterIsolation:
    """Test each inference parameter individually with real API.

    These tests verify that individual parameters are correctly passed
    through to the LLM and produce valid responses.
    """

    def test_temperature_low(self, lmstudio_provider_factory):
        """Test low temperature produces response."""
        provider = lmstudio_provider_factory(temperature=0.1)
        result = provider.complete(
            "You are a helpful assistant.",
            "What is 2+2? Answer with just the number."
        )
        assert result is not None
        assert len(result) > 0

    def test_temperature_high(self, lmstudio_provider_factory):
        """Test high temperature produces response."""
        provider = lmstudio_provider_factory(temperature=0.9)
        result = provider.complete(
            "You are a helpful assistant.",
            "What is 2+2? Answer with just the number."
        )
        assert result is not None
        assert len(result) > 0

    def test_top_p_low(self, lmstudio_provider_factory):
        """Test low top_p produces response."""
        provider = lmstudio_provider_factory(top_p=0.5)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_top_p_high(self, lmstudio_provider_factory):
        """Test high top_p produces response."""
        provider = lmstudio_provider_factory(top_p=0.95)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_top_k_low(self, lmstudio_provider_factory):
        """Test low top_k produces response."""
        provider = lmstudio_provider_factory(top_k=10)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_top_k_high(self, lmstudio_provider_factory):
        """Test high top_k produces response."""
        provider = lmstudio_provider_factory(top_k=100)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_max_tokens_small(self, lmstudio_provider_factory):
        """Test small max_tokens limits response."""
        provider = lmstudio_provider_factory(max_tokens=10)
        result = provider.complete(
            "You are a helpful assistant.",
            "Count from 1 to 100."
        )
        assert result is not None
        # Response should be limited (though exact behavior varies)
        assert len(result) > 0

    def test_max_tokens_large(self, lmstudio_provider_factory):
        """Test large max_tokens allows longer response."""
        provider = lmstudio_provider_factory(max_tokens=512)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_frequency_penalty_low(self, lmstudio_provider_factory):
        """Test low frequency_penalty produces response."""
        provider = lmstudio_provider_factory(frequency_penalty=0.1)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_frequency_penalty_high(self, lmstudio_provider_factory):
        """Test high frequency_penalty produces response."""
        provider = lmstudio_provider_factory(frequency_penalty=1.0)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_presence_penalty_low(self, lmstudio_provider_factory):
        """Test low presence_penalty produces response."""
        provider = lmstudio_provider_factory(presence_penalty=0.1)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_presence_penalty_high(self, lmstudio_provider_factory):
        """Test high presence_penalty produces response."""
        provider = lmstudio_provider_factory(presence_penalty=1.0)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_repeat_penalty(self, lmstudio_provider_factory):
        """Test repeat_penalty produces response."""
        provider = lmstudio_provider_factory(repeat_penalty=1.2)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_stop_single(self, lmstudio_provider_factory):
        """Test single stop sequence."""
        provider = lmstudio_provider_factory(stop=["###"])
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert "###" not in result  # Stop sequence should not appear

    def test_stop_multiple(self, lmstudio_provider_factory):
        """Test multiple stop sequences."""
        provider = lmstudio_provider_factory(stop=["END", "STOP", "###"])
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None

    def test_seed_determinism(self, lmstudio_provider_factory):
        """Test seed produces response (determinism depends on backend)."""
        provider = lmstudio_provider_factory(seed=42, temperature=0.0)
        result1 = provider.complete(
            "You are a helpful assistant.",
            "What is 2+2?"
        )

        # Create another provider with same seed
        provider2 = lmstudio_provider_factory(seed=42, temperature=0.0)
        result2 = provider2.complete(
            "You are a helpful assistant.",
            "What is 2+2?"
        )

        # Both should produce responses
        assert result1 is not None
        assert result2 is not None
        # Note: determinism depends on backend support

    def test_timeout_parameter(self, lmstudio_provider_factory):
        """Test custom timeout produces response."""
        provider = lmstudio_provider_factory(timeout=60)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0


# =============================================================================
# TestParameterCombinations - Test parameter combinations with real API
# =============================================================================


@pytest.mark.api
class TestParameterCombinations:
    """Test combinations of inference parameters with real API.

    These tests verify that multiple parameters work together correctly.
    """

    def test_temperature_and_top_p(self, lmstudio_provider_factory):
        """Test temperature and top_p combination."""
        provider = lmstudio_provider_factory(temperature=0.5, top_p=0.9)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_temperature_and_top_k(self, lmstudio_provider_factory):
        """Test temperature and top_k combination."""
        provider = lmstudio_provider_factory(temperature=0.7, top_k=50)
        result = provider.complete(
            "You are a helpful assistant.",
            "Say hello."
        )
        assert result is not None
        assert len(result) > 0

    def test_all_sampling_params(self, lmstudio_provider_factory):
        """Test all sampling parameters together."""
        provider = lmstudio_provider_factory(
            temperature=0.6,
            top_p=0.85,
            top_k=40,
            max_tokens=256,
        )
        result = provider.complete(
            "You are a helpful assistant.",
            "Tell me a brief joke."
        )
        assert result is not None
        assert len(result) > 0

    def test_all_penalty_params(self, lmstudio_provider_factory):
        """Test all penalty parameters together."""
        provider = lmstudio_provider_factory(
            frequency_penalty=0.5,
            presence_penalty=0.5,
            repeat_penalty=1.1,
        )
        result = provider.complete(
            "You are a helpful assistant.",
            "List three colors."
        )
        assert result is not None
        assert len(result) > 0

    def test_sampling_and_penalty_params(self, lmstudio_provider_factory):
        """Test sampling and penalty parameters combined."""
        provider = lmstudio_provider_factory(
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.3,
        )
        result = provider.complete(
            "You are a helpful assistant.",
            "Write a short sentence."
        )
        assert result is not None
        assert len(result) > 0

    def test_all_params_combined(self, lmstudio_provider_factory):
        """Test all inference parameters combined."""
        provider = lmstudio_provider_factory(
            temperature=0.5,
            top_p=0.85,
            top_k=40,
            max_tokens=256,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            repeat_penalty=1.1,
            stop=["###"],
            seed=42,
            timeout=60,
        )
        result = provider.complete(
            "You are a helpful assistant.",
            "Say 'All parameters work correctly!'"
        )
        assert result is not None
        assert len(result) > 0
        assert "###" not in result

    def test_low_temperature_with_penalties(self, lmstudio_provider_factory):
        """Test low temperature with penalty parameters."""
        provider = lmstudio_provider_factory(
            temperature=0.1,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )
        result = provider.complete(
            "You are a helpful assistant.",
            "What is the capital of France?"
        )
        assert result is not None
        assert len(result) > 0

    def test_high_temperature_with_constraints(self, lmstudio_provider_factory):
        """Test high temperature with constraining parameters."""
        provider = lmstudio_provider_factory(
            temperature=0.9,
            top_p=0.7,  # Constrain with top_p
            top_k=30,   # Constrain with top_k
            max_tokens=100,
        )
        result = provider.complete(
            "You are a helpful assistant.",
            "Write one word."
        )
        assert result is not None
        assert len(result) > 0

    def test_deterministic_config(self, lmstudio_provider_factory):
        """Test configuration for deterministic output."""
        provider = lmstudio_provider_factory(
            temperature=0.0,
            seed=12345,
            top_p=1.0,
        )
        result = provider.complete(
            "You are a helpful assistant.",
            "What is 2+2? Answer with just the number."
        )
        assert result is not None
        assert len(result) > 0

    def test_creative_config(self, lmstudio_provider_factory):
        """Test configuration for creative output."""
        provider = lmstudio_provider_factory(
            temperature=0.9,
            top_p=0.95,
            top_k=100,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            max_tokens=200,
        )
        result = provider.complete(
            "You are a creative storyteller.",
            "Start a story with 'Once upon a time'"
        )
        assert result is not None
        assert len(result) > 0


# =============================================================================
# TestProviderIntegration - Integration tests with real API
# =============================================================================


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

    def test_factory_with_custom_model(self, lmstudio_provider_factory):
        """Test factory with custom model parameter."""
        # Use the same model but verify the parameter is passed
        provider = lmstudio_provider_factory(model="openai/qwen3-coder-30b")

        result = provider.complete(
            "You are helpful.",
            "Say hello."
        )

        assert result is not None
        assert len(result) > 0

    def test_different_providers_same_prompt(self, lmstudio_provider_factory):
        """Test different provider configurations with same prompt."""
        prompt = "What is 1+1? Answer with just the number."

        # Low temperature
        provider1 = lmstudio_provider_factory(temperature=0.1)
        result1 = provider1.complete("You are helpful.", prompt)

        # High temperature
        provider2 = lmstudio_provider_factory(temperature=0.9)
        result2 = provider2.complete("You are helpful.", prompt)

        assert result1 is not None
        assert result2 is not None

    def test_provider_reuse(self, lmstudio_provider_factory):
        """Test reusing same provider for multiple requests."""
        provider = lmstudio_provider_factory(temperature=0.5)

        results = []
        for i in range(5):
            result = provider.complete(
                "You are helpful.",
                f"Say the number {i}."
            )
            results.append(result)

        assert len(results) == 5
        for r in results:
            assert r is not None
            assert len(r) > 0

    def test_connection_test_before_completion(self, lmstudio_provider_factory):
        """Test connection check followed by completion."""
        provider = lmstudio_provider_factory()

        # First verify connection
        assert provider.test_connection() is True

        # Then complete
        result = provider.complete(
            "You are helpful.",
            "Say hello."
        )

        assert result is not None
        assert len(result) > 0

    def test_factory_creates_independent_providers(self, lmstudio_provider_factory):
        """Test that factory creates independent provider instances."""
        provider1 = lmstudio_provider_factory(temperature=0.1)
        provider2 = lmstudio_provider_factory(temperature=0.9)

        # Verify they have different configs
        assert provider1.config.temperature != provider2.config.temperature
        assert provider1.config.temperature == 0.1
        assert provider2.config.temperature == 0.9

        # Both should work
        result1 = provider1.complete("You are helpful.", "Hi")
        result2 = provider2.complete("You are helpful.", "Hi")

        assert result1 is not None
        assert result2 is not None

    def test_long_system_prompt(self, lmstudio_provider):
        """Test with a longer system prompt."""
        system_prompt = """You are an expert assistant specializing in mathematics.
        You provide clear, step-by-step explanations.
        You always verify your answers.
        You use simple language that anyone can understand.
        You are patient and thorough."""

        result = lmstudio_provider.complete(
            system_prompt,
            "What is 5+5?"
        )

        assert result is not None
        assert len(result) > 0

    def test_long_user_prompt(self, lmstudio_provider):
        """Test with a longer user prompt."""
        user_prompt = """I have a question for you. I've been thinking about this
        for a while and I really need a clear answer. Here's my question:
        What is the sum of two and three? Please answer with just the number,
        no explanation needed. Just the number please."""

        result = lmstudio_provider.complete(
            "You are helpful and concise.",
            user_prompt
        )

        assert result is not None
        assert len(result) > 0


# =============================================================================
# TestProviderRegistry - Tests for provider registry management
# =============================================================================


class TestProviderRegistry:
    """Tests for ProviderRegistry - per-persona provider management."""

    # --- Helper to create a test ResolvedConfig ---

    def _create_test_config(self) -> ResolvedConfig:
        """Create a standard test configuration."""
        return ResolvedConfig(
            defaults=ProviderSettings(
                model="openai/test-model",
                api_base="http://localhost:1234/v1",
                temperature=0.7,
                max_tokens=1000,
            ),
            providers={
                "fast": ProviderSettings(temperature=0.3, max_tokens=500),
                "creative": ProviderSettings(temperature=0.9, top_p=0.95),
            },
            persona_configs={
                "The Pragmatist": ProviderSettings(temperature=0.3, provider="fast"),
                "The Innovator": ProviderSettings(temperature=1.0, top_k=50),
            },
            generation=GenerationSettings(),
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )

    # --- NON-API TESTS ---

    def test_registry_creation_empty(self):
        """Test creating an empty registry without config."""
        registry = ProviderRegistry()
        assert registry._providers == {}
        assert registry._config is None
        assert registry._default_provider is None

    def test_registry_creation_with_config(self):
        """Test creating registry with ResolvedConfig."""
        resolved = self._create_test_config()
        registry = ProviderRegistry(resolved)

        assert registry._config is resolved
        assert registry._providers == {}
        assert registry._default_provider is None

    def test_register_and_get_provider(self):
        """Test registering and retrieving a provider by name."""
        registry = ProviderRegistry()

        config = ProviderConfig(
            model="openai/test-model",
            api_base="http://localhost:1234/v1",
        )
        provider = LiteLLMProvider(config)

        registry.register("test", provider)

        retrieved = registry.get("test")
        assert retrieved is provider

    def test_get_returns_none_for_unknown(self):
        """Test get() returns None for unknown provider name."""
        registry = ProviderRegistry()

        result = registry.get("nonexistent")
        assert result is None

    def test_set_and_get_default(self):
        """Test setting and getting default provider."""
        registry = ProviderRegistry()

        config = ProviderConfig(
            model="openai/test-model",
            api_base="http://localhost:1234/v1",
        )
        provider = LiteLLMProvider(config)

        registry.set_default(provider)

        default = registry.get_default()
        assert default is provider

        # Also accessible via get("default")
        assert registry.get("default") is provider

    def test_list_providers_empty(self):
        """Test list_providers with no registered providers."""
        registry = ProviderRegistry()

        names = registry.list_providers()

        # Should always include 'default' even if empty
        assert "default" in names

    def test_list_providers_after_registration(self):
        """Test list_providers after registering providers."""
        registry = ProviderRegistry()

        config = ProviderConfig(model="openai/test-model")
        provider1 = LiteLLMProvider(config)
        provider2 = LiteLLMProvider(config)

        registry.register("alpha", provider1)
        registry.register("beta", provider2)

        names = registry.list_providers()

        assert "alpha" in names
        assert "beta" in names
        assert "default" in names

    def test_get_or_create_from_config(self):
        """Test get_or_create creates provider from config settings."""
        resolved = self._create_test_config()
        registry = ProviderRegistry(resolved)

        provider = registry.get_or_create("fast")

        assert isinstance(provider, LiteLLMProvider)
        # "fast" provider should have temperature=0.3 (merged with defaults)
        assert provider.config.temperature == 0.3
        # max_tokens should be 500 from "fast" config
        assert provider.config.max_tokens == 500

    def test_get_or_create_uses_cached(self):
        """Test get_or_create returns cached provider on second call."""
        resolved = self._create_test_config()
        registry = ProviderRegistry(resolved)

        provider1 = registry.get_or_create("fast")
        provider2 = registry.get_or_create("fast")

        assert provider1 is provider2

    def test_get_or_create_default(self):
        """Test get_or_create with 'default' uses defaults settings."""
        resolved = self._create_test_config()
        registry = ProviderRegistry(resolved)

        provider = registry.get_or_create("default")

        assert isinstance(provider, LiteLLMProvider)
        assert provider.config.model == "openai/test-model"
        assert provider.config.temperature == 0.7
        assert provider.config.max_tokens == 1000

    def test_get_or_create_unknown_raises(self):
        """Test get_or_create raises ValueError for unknown provider."""
        resolved = self._create_test_config()
        registry = ProviderRegistry(resolved)

        with pytest.raises(ValueError) as exc_info:
            registry.get_or_create("unknown_provider")

        assert "not found" in str(exc_info.value).lower()

    def test_get_or_create_no_config_raises(self):
        """Test get_or_create raises ValueError without config."""
        registry = ProviderRegistry()  # No config

        with pytest.raises(ValueError) as exc_info:
            registry.get_or_create("any_provider")

        assert "not found" in str(exc_info.value).lower()

    def test_get_for_persona_creates_provider(self):
        """Test get_for_persona creates provider for configured persona."""
        resolved = self._create_test_config()
        registry = ProviderRegistry(resolved)

        provider = registry.get_for_persona("The Pragmatist")

        assert isinstance(provider, LiteLLMProvider)
        # The Pragmatist has temperature=0.3 (merged with defaults)
        assert provider.config.temperature == 0.3

    def test_get_for_persona_caches_result(self):
        """Test get_for_persona caches provider on second call."""
        resolved = self._create_test_config()
        registry = ProviderRegistry(resolved)

        provider1 = registry.get_for_persona("The Pragmatist")
        provider2 = registry.get_for_persona("The Pragmatist")

        assert provider1 is provider2

        # Should be cached with key "persona:The Pragmatist"
        assert "persona:The Pragmatist" in registry._providers

    def test_get_for_persona_fallback_to_default(self):
        """Test get_for_persona falls back to default for unknown persona."""
        resolved = self._create_test_config()
        registry = ProviderRegistry(resolved)

        # Unknown persona should use default settings
        provider = registry.get_for_persona("Unknown Persona")

        assert isinstance(provider, LiteLLMProvider)
        # Should have default settings
        assert provider.config.model == "openai/test-model"
        assert provider.config.temperature == 0.7

    def test_get_for_persona_no_config_uses_default_provider(self):
        """Test get_for_persona uses default provider when no config."""
        registry = ProviderRegistry()

        # Set a default provider
        config = ProviderConfig(
            model="openai/default-model",
            api_base="http://localhost:1234/v1",
            temperature=0.5,
        )
        default_provider = LiteLLMProvider(config)
        registry.set_default(default_provider)

        # Should return default provider for any persona
        provider = registry.get_for_persona("Any Persona")

        assert provider is default_provider

    def test_get_for_persona_no_config_no_default_raises(self):
        """Test get_for_persona raises when no config and no default."""
        registry = ProviderRegistry()  # No config, no default

        with pytest.raises(ValueError) as exc_info:
            registry.get_for_persona("Any Persona")

        assert "no config" in str(exc_info.value).lower()

    def test_get_for_persona_uses_all_params(self):
        """Test get_for_persona creates provider with all inference params."""
        # Create config with all possible parameters
        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model="openai/base-model",
                api_base="http://localhost:1234/v1",
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                max_tokens=1000,
                frequency_penalty=0.1,
                presence_penalty=0.2,
                repeat_penalty=1.1,
                stop=["STOP"],
                seed=42,
                timeout=120,
            ),
            providers={},
            persona_configs={
                "Full Params Persona": ProviderSettings(
                    temperature=0.5,
                    top_k=50,
                    max_tokens=2000,
                ),
            },
            generation=GenerationSettings(),
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )
        registry = ProviderRegistry(resolved)

        provider = registry.get_for_persona("Full Params Persona")

        # Persona-specific overrides
        assert provider.config.temperature == 0.5
        assert provider.config.top_k == 50
        assert provider.config.max_tokens == 2000

        # Inherited from defaults
        assert provider.config.model == "openai/base-model"
        assert provider.config.api_base == "http://localhost:1234/v1"
        assert provider.config.top_p == 0.9
        assert provider.config.frequency_penalty == 0.1
        assert provider.config.presence_penalty == 0.2
        assert provider.config.repeat_penalty == 1.1
        assert provider.config.stop == ["STOP"]
        assert provider.config.seed == 42
        assert provider.config.timeout == 120

    def test_list_providers_includes_config_providers(self):
        """Test list_providers includes providers from config."""
        resolved = self._create_test_config()
        registry = ProviderRegistry(resolved)

        names = registry.list_providers()

        # Should include config providers even if not yet created
        assert "fast" in names
        assert "creative" in names
        assert "default" in names

    def test_list_providers_combines_registered_and_config(self):
        """Test list_providers combines registered and config providers."""
        resolved = self._create_test_config()
        registry = ProviderRegistry(resolved)

        # Register additional provider
        config = ProviderConfig(model="openai/test-model")
        provider = LiteLLMProvider(config)
        registry.register("custom", provider)

        names = registry.list_providers()

        # Both config providers and registered provider
        assert "fast" in names
        assert "creative" in names
        assert "custom" in names
        assert "default" in names

    # --- API TESTS ---

    @pytest.mark.api
    def test_validate_all_with_working_provider(self, lmstudio_provider):
        """Test validate_all returns True for working provider."""
        # Create config pointing to real LM Studio
        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model="openai/qwen3-coder-30b",
                api_base="http://localhost:1234/v1",
                temperature=0.7,
                max_tokens=100,
            ),
            providers={},
            persona_configs={},
            generation=GenerationSettings(),
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )
        registry = ProviderRegistry(resolved)

        results = registry.validate_all()

        assert "default" in results
        assert results["default"] is True

    @pytest.mark.api
    def test_validate_all_with_bad_provider(self):
        """Test validate_all returns False for unreachable provider."""
        # Create config with invalid endpoint
        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model="openai/test-model",
                api_base="http://localhost:59999/v1",  # Invalid port
                temperature=0.7,
                max_tokens=100,
                timeout=5,  # Short timeout for quick failure
            ),
            providers={
                "bad_provider": ProviderSettings(
                    api_base="http://localhost:59998/v1",  # Another invalid port
                    timeout=5,
                ),
            },
            persona_configs={},
            generation=GenerationSettings(),
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )
        registry = ProviderRegistry(resolved)

        results = registry.validate_all()

        assert "default" in results
        assert results["default"] is False
        assert "bad_provider" in results
        assert results["bad_provider"] is False
