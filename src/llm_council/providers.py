"""LLM Provider implementations using LiteLLM."""

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import litellm

# Suppress Pydantic serialization warnings from LiteLLM
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    # Sampling
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: int = 1024
    # Repetition control
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repeat_penalty: Optional[float] = None  # LM Studio extension
    # Control
    stop: Optional[list[str]] = None
    seed: Optional[int] = None
    timeout: int = 120


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the provider is accessible."""
        pass


class LiteLLMProvider(LLMProvider):
    """LiteLLM-based provider supporting multiple backends."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        # Configure LiteLLM
        if config.api_base:
            # For local models via LM Studio, use openai/ prefix
            litellm.api_base = config.api_base
        if config.api_key:
            litellm.api_key = config.api_key

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a completion using LiteLLM."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Build kwargs with required params
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "drop_params": True,  # Allow unsupported params to be dropped by provider
        }

        # Add api_base if specified (for local LM Studio)
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        # Add api_key if specified
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

        # Add optional sampling params if set
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            kwargs["top_k"] = self.config.top_k

        # Add optional repetition control params if set
        if self.config.frequency_penalty is not None:
            kwargs["frequency_penalty"] = self.config.frequency_penalty
        if self.config.presence_penalty is not None:
            kwargs["presence_penalty"] = self.config.presence_penalty
        if self.config.repeat_penalty is not None:
            kwargs["repeat_penalty"] = self.config.repeat_penalty

        # Add optional control params if set
        if self.config.stop is not None:
            kwargs["stop"] = self.config.stop
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed

        response = litellm.completion(**kwargs)
        return response.choices[0].message.content

    def test_connection(self) -> bool:
        """Test connection by making a simple request."""
        try:
            result = self.complete(
                system_prompt="You are a helpful assistant.",
                user_prompt="Respond with only the word 'OK'.",
            )
            return len(result) > 0
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


def create_provider(
    provider_type: str = "litellm",
    model: str = "openai/qwen/qwen3-coder-30b",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: int = 1024,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    stop: Optional[list[str]] = None,
    seed: Optional[int] = None,
    timeout: int = 120,
) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        provider_type: Type of provider ("litellm" or "openai")
        model: Model name. For local LM Studio, use "openai/<model-name>"
        api_base: Base URL for API (e.g., "http://localhost:1234/v1" for LM Studio)
        api_key: API key if required
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Maximum tokens in response
        frequency_penalty: Frequency penalty for repetition control
        presence_penalty: Presence penalty for repetition control
        repeat_penalty: Repeat penalty (LM Studio extension)
        stop: Stop sequences
        seed: Random seed for reproducibility
        timeout: Request timeout in seconds

    Returns:
        LLMProvider instance
    """
    config = ProviderConfig(
        model=model,
        api_base=api_base,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repeat_penalty=repeat_penalty,
        stop=stop,
        seed=seed,
        timeout=timeout,
    )

    if provider_type == "litellm":
        return LiteLLMProvider(config)
    else:
        # Default to LiteLLM for everything
        return LiteLLMProvider(config)


# Preset configurations for common setups
PRESETS = {
    "lmstudio": {
        "provider_type": "litellm",
        "api_base": "http://localhost:1234/v1",
        "api_key": "lm-studio",  # LM Studio doesn't need a real key
    },
    "openai": {
        "provider_type": "litellm",
        "model": "gpt-4o",
    },
    "openai-mini": {
        "provider_type": "litellm",
        "model": "gpt-4o-mini",
    },
    "anthropic": {
        "provider_type": "litellm",
        "model": "claude-3-opus-20240229",
    },
    "ollama": {
        "provider_type": "litellm",
        "api_base": "http://localhost:11434",
    },
}


class ProviderRegistry:
    """Registry for managing multiple LLM providers.

    Supports per-persona provider resolution with config inheritance.
    """

    def __init__(self, resolved_config=None):
        """Initialize registry with optional resolved configuration.

        Args:
            resolved_config: ResolvedConfig from ConfigManager.resolve()
        """
        self._providers: dict[str, LLMProvider] = {}
        self._config = resolved_config
        self._default_provider: Optional[LLMProvider] = None

    def register(self, name: str, provider: LLMProvider):
        """Register a named provider."""
        self._providers[name] = provider

    def get(self, name: str) -> Optional[LLMProvider]:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_or_create(self, name: str) -> LLMProvider:
        """Get existing provider or create from config."""
        if name in self._providers:
            return self._providers[name]

        if self._config is None:
            raise ValueError(f"Provider '{name}' not found and no config available")

        # Check named providers in config
        if name in self._config.providers:
            settings = self._config.defaults.merge_with(self._config.providers[name])
        elif name == 'default':
            settings = self._config.defaults
        else:
            raise ValueError(f"Provider '{name}' not found in config")

        provider = create_provider(
            model=settings.model or "openai/qwen/qwen3-coder-30b",
            api_base=settings.api_base,
            api_key=settings.api_key,
            temperature=settings.temperature or 0.7,
            top_p=settings.top_p,
            top_k=settings.top_k,
            max_tokens=settings.max_tokens or 1024,
            frequency_penalty=settings.frequency_penalty,
            presence_penalty=settings.presence_penalty,
            repeat_penalty=settings.repeat_penalty,
            stop=settings.stop,
            seed=settings.seed,
            timeout=settings.timeout or 120,
        )
        self._providers[name] = provider
        return provider

    def get_for_persona(self, persona_name: str) -> LLMProvider:
        """Get provider for a specific persona.

        Uses merge inheritance: defaults -> named provider -> persona overrides.
        """
        if self._config is None:
            if self._default_provider:
                return self._default_provider
            raise ValueError("No config or default provider available")

        # Check cache first
        cache_key = f"persona:{persona_name}"
        if cache_key in self._providers:
            return self._providers[cache_key]

        # Get merged settings for persona
        from .config import ConfigManager
        manager = ConfigManager()
        settings = manager.get_provider_for_persona(persona_name, self._config)
        defaults = self._config.defaults

        # Create provider with all inference parameters
        provider = create_provider(
            model=settings.model or defaults.model or "openai/qwen/qwen3-coder-30b",
            api_base=settings.api_base or defaults.api_base,
            api_key=settings.api_key or defaults.api_key,
            temperature=settings.temperature or defaults.temperature or 0.7,
            top_p=settings.top_p if settings.top_p is not None else defaults.top_p,
            top_k=settings.top_k if settings.top_k is not None else defaults.top_k,
            max_tokens=settings.max_tokens or defaults.max_tokens or 1024,
            frequency_penalty=settings.frequency_penalty if settings.frequency_penalty is not None else defaults.frequency_penalty,
            presence_penalty=settings.presence_penalty if settings.presence_penalty is not None else defaults.presence_penalty,
            repeat_penalty=settings.repeat_penalty if settings.repeat_penalty is not None else defaults.repeat_penalty,
            stop=settings.stop if settings.stop is not None else defaults.stop,
            seed=settings.seed if settings.seed is not None else defaults.seed,
            timeout=settings.timeout or defaults.timeout or 120,
        )

        self._providers[cache_key] = provider
        return provider

    def set_default(self, provider: LLMProvider):
        """Set the default provider."""
        self._default_provider = provider
        self._providers['default'] = provider

    def get_default(self) -> LLMProvider:
        """Get the default provider."""
        if self._default_provider:
            return self._default_provider
        return self.get_or_create('default')

    def validate_all(self) -> dict[str, bool]:
        """Validate all configured providers.

        Returns dict of provider_name -> is_valid.
        """
        results = {}

        # Validate default
        try:
            default = self.get_default()
            results['default'] = default.test_connection()
        except Exception:
            results['default'] = False

        # Validate named providers
        if self._config:
            for name in self._config.providers:
                try:
                    provider = self.get_or_create(name)
                    results[name] = provider.test_connection()
                except Exception:
                    results[name] = False

        return results

    def list_providers(self) -> list[str]:
        """List all available provider names."""
        names = set(self._providers.keys())
        if self._config:
            names.update(self._config.providers.keys())
        names.add('default')
        return sorted(names)


def create_provider_from_settings(settings) -> LLMProvider:
    """Create provider from ProviderSettings object.

    Args:
        settings: ProviderSettings from config module
    """
    return create_provider(
        model=settings.model or "openai/qwen/qwen3-coder-30b",
        api_base=settings.api_base,
        api_key=settings.api_key,
        temperature=settings.temperature or 0.7,
        top_p=settings.top_p,
        top_k=settings.top_k,
        max_tokens=settings.max_tokens or 1024,
        frequency_penalty=settings.frequency_penalty,
        presence_penalty=settings.presence_penalty,
        repeat_penalty=settings.repeat_penalty,
        stop=settings.stop,
        seed=settings.seed,
        timeout=settings.timeout or 120,
    )
