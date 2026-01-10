"""LLM Provider implementations using LiteLLM."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import litellm


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
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

        # Build kwargs
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

        # Add api_base if specified (for local LM Studio)
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        # Add api_key if specified
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key

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
    model: str = "openai/qwen3-coder-30b",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        provider_type: Type of provider ("litellm" or "openai")
        model: Model name. For local LM Studio, use "openai/<model-name>"
        api_base: Base URL for API (e.g., "http://localhost:1234/v1" for LM Studio)
        api_key: API key if required
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        LLMProvider instance
    """
    config = ProviderConfig(
        model=model,
        api_base=api_base,
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
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
}
