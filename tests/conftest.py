"""Pytest configuration and fixtures for LLM Council tests.

POLICY: NO MOCKED API TESTS
All API tests MUST reach actual LM Studio endpoint.
See CLAUDE.md for rationale.
"""

import socket
import pytest
from typing import Generator

from llm_council.providers import create_provider, LiteLLMProvider, ProviderConfig


# LM Studio configuration
LMSTUDIO_HOST = "localhost"
LMSTUDIO_PORT = 1234
LMSTUDIO_API_BASE = f"http://{LMSTUDIO_HOST}:{LMSTUDIO_PORT}/v1"
LMSTUDIO_MODEL = "openai/qwen/qwen3-coder-30b"  # Any model loaded in LM Studio


def is_lmstudio_running() -> bool:
    """Check if LM Studio server is accepting connections.

    Returns:
        True if LM Studio is running on localhost:1234
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((LMSTUDIO_HOST, LMSTUDIO_PORT))
        sock.close()
        return result == 0
    except Exception:
        return False


# Register custom pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "api: marks tests that require LM Studio API (deselect with '-m \"not api\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip API tests if LM Studio is not running.

    Per CLAUDE.md policy: Tests should FAIL if LM Studio is unavailable in CI.
    But for local development, we skip with a clear warning.
    """
    import os

    # In CI, don't skip - let tests fail to enforce LM Studio requirement
    if os.environ.get("CI"):
        return

    if not is_lmstudio_running():
        skip_api = pytest.mark.skip(
            reason="LM Studio not running on localhost:1234. Start LM Studio to run API tests."
        )
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip_api)


@pytest.fixture(scope="session")
def lmstudio_available() -> bool:
    """Check if LM Studio is available (session-scoped for efficiency)."""
    return is_lmstudio_running()


@pytest.fixture
def stub_provider() -> LiteLLMProvider:
    """Create a provider instance without testing connection.

    Use this for tests that need a provider object but don't make API calls.
    For tests that actually call the API, use lmstudio_provider instead.
    """
    config = ProviderConfig(
        model=LMSTUDIO_MODEL,
        api_base=LMSTUDIO_API_BASE,
        temperature=0.7,
        max_tokens=1024,
        timeout=120,
    )
    return LiteLLMProvider(config)


@pytest.fixture(scope="session")
def lmstudio_provider() -> Generator[LiteLLMProvider, None, None]:
    """Create a real LM Studio provider for API tests.

    This is session-scoped to avoid creating new connections for each test.
    Tests using this fixture will be marked as API tests.

    Raises:
        pytest.skip: If LM Studio is not running (local dev only)
        AssertionError: If LM Studio is not running (CI environment)
    """
    import os

    if not is_lmstudio_running():
        if os.environ.get("CI"):
            pytest.fail("LM Studio must be running for API tests in CI")
        else:
            pytest.skip("LM Studio not running on localhost:1234")

    config = ProviderConfig(
        model=LMSTUDIO_MODEL,
        api_base=LMSTUDIO_API_BASE,
        temperature=0.7,
        max_tokens=1024,
        timeout=120,
    )
    provider = LiteLLMProvider(config)

    # Verify connection works
    if not provider.test_connection():
        pytest.fail(f"LM Studio connection test failed at {LMSTUDIO_API_BASE}")

    yield provider


@pytest.fixture
def lmstudio_provider_factory():
    """Factory fixture to create fresh providers with custom config.

    Supports all inference parameters:
    - model, api_base, api_key: Connection settings
    - temperature, top_p, top_k, max_tokens: Sampling parameters
    - frequency_penalty, presence_penalty, repeat_penalty: Repetition control
    - stop, seed, timeout: Control parameters

    Usage:
        def test_something(lmstudio_provider_factory):
            provider = lmstudio_provider_factory(temperature=0.5, top_p=0.9)
    """
    import os

    if not is_lmstudio_running():
        if os.environ.get("CI"):
            pytest.fail("LM Studio must be running for API tests in CI")
        else:
            pytest.skip("LM Studio not running on localhost:1234")

    def _create_provider(**kwargs) -> LiteLLMProvider:
        config = ProviderConfig(
            model=kwargs.get("model", LMSTUDIO_MODEL),
            api_base=kwargs.get("api_base", LMSTUDIO_API_BASE),
            api_key=kwargs.get("api_key"),
            # Sampling parameters
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
            max_tokens=kwargs.get("max_tokens", 1024),
            # Repetition control
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            repeat_penalty=kwargs.get("repeat_penalty"),
            # Control parameters
            stop=kwargs.get("stop"),
            seed=kwargs.get("seed"),
            timeout=kwargs.get("timeout", 120),
        )
        return LiteLLMProvider(config)

    return _create_provider


@pytest.fixture
def simple_personas():
    """Return a minimal set of 3 personas for testing."""
    from llm_council.models import DEFAULT_PERSONAS
    return DEFAULT_PERSONAS[:3]


@pytest.fixture
def council_engine_factory(lmstudio_provider):
    """Factory to create CouncilEngine with real provider.

    Usage:
        def test_council(council_engine_factory):
            engine = council_engine_factory(max_rounds=2)
    """
    from llm_council.council import CouncilEngine
    from llm_council.models import ConsensusType

    def _create_engine(**kwargs) -> CouncilEngine:
        return CouncilEngine(
            provider=kwargs.get("provider", lmstudio_provider),
            consensus_type=kwargs.get("consensus_type", ConsensusType.MAJORITY),
            max_rounds=kwargs.get("max_rounds", 2),
            stalemate_threshold=kwargs.get("stalemate_threshold", 3),
        )

    return _create_engine
