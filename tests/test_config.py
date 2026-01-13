"""Tests for configuration management module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from llm_council.config import (
    ConfigManager,
    ConfigSchema,
    ProviderSettings,
    GenerationSettings,
    CouncilSettings,
    PersistenceSettings,
    ResolvedConfig,
    get_user_config_dir,
    get_user_config_path,
    get_project_config_path,
    resolve_env_vars,
    load_config,
    save_config,
    get_default_config,
)


class TestResolveEnvVars:
    """Tests for environment variable resolution."""

    def test_resolve_simple_env_var(self):
        """Test resolving a simple env var."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = resolve_env_vars("${TEST_VAR}")
            assert result == "test_value"

    def test_resolve_env_var_in_string(self):
        """Test resolving env var embedded in string."""
        with patch.dict(os.environ, {"API_KEY": "sk-123"}):
            result = resolve_env_vars("Bearer ${API_KEY}")
            assert result == "Bearer sk-123"

    def test_missing_env_var_keeps_original(self):
        """Test that missing env vars keep original syntax."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the var doesn't exist
            os.environ.pop("NONEXISTENT_VAR", None)
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = resolve_env_vars("${NONEXISTENT_VAR}")
                assert result == "${NONEXISTENT_VAR}"
                assert len(w) == 1
                assert "NONEXISTENT_VAR" in str(w[0].message)

    def test_resolve_nested_dict(self):
        """Test resolving env vars in nested dict."""
        with patch.dict(os.environ, {"KEY": "value"}):
            data = {"outer": {"inner": "${KEY}"}}
            result = resolve_env_vars(data)
            assert result == {"outer": {"inner": "value"}}

    def test_resolve_list(self):
        """Test resolving env vars in list."""
        with patch.dict(os.environ, {"VAR1": "a", "VAR2": "b"}):
            data = ["${VAR1}", "${VAR2}"]
            result = resolve_env_vars(data)
            assert result == ["a", "b"]


class TestProviderSettings:
    """Tests for ProviderSettings model."""

    def test_create_with_defaults(self):
        """Test creating with default values."""
        settings = ProviderSettings()
        assert settings.model is None
        assert settings.temperature is None

    def test_create_with_values(self):
        """Test creating with explicit values."""
        settings = ProviderSettings(
            model="gpt-4o",
            temperature=0.8,
            max_tokens=2048,
        )
        assert settings.model == "gpt-4o"
        assert settings.temperature == 0.8
        assert settings.max_tokens == 2048

    def test_temperature_validation(self):
        """Test temperature range validation."""
        # Valid
        settings = ProviderSettings(temperature=1.5)
        assert settings.temperature == 1.5

        # Invalid - too high
        with pytest.raises(ValueError):
            ProviderSettings(temperature=3.0)

        # Invalid - too low
        with pytest.raises(ValueError):
            ProviderSettings(temperature=-0.5)

    def test_merge_with(self):
        """Test merging two settings objects."""
        base = ProviderSettings(model="gpt-3.5", temperature=0.5)
        override = ProviderSettings(temperature=0.9, max_tokens=1024)

        merged = base.merge_with(override)

        assert merged.model == "gpt-3.5"  # From base
        assert merged.temperature == 0.9  # From override
        assert merged.max_tokens == 1024  # From override

    def test_resolve_env_vars(self):
        """Test resolving env vars in settings."""
        with patch.dict(os.environ, {"MY_API_KEY": "sk-secret"}):
            settings = ProviderSettings(api_key="${MY_API_KEY}")
            resolved = settings.resolve_env_vars()
            assert resolved.api_key == "sk-secret"

    def test_plaintext_api_key_warning(self):
        """Test warning for plaintext API keys."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProviderSettings(api_key="sk-this-is-a-very-long-api-key-1234567890")
            assert len(w) == 1
            assert "plaintext" in str(w[0].message).lower()

    def test_env_var_api_key_no_warning(self):
        """Test no warning for env var referenced API keys."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProviderSettings(api_key="${OPENAI_API_KEY}")
            assert len(w) == 0

    def test_top_p_validation(self):
        """Test top_p range validation (0.0 to 1.0)."""
        # Valid values
        for value in [0.0, 0.5, 1.0]:
            settings = ProviderSettings(top_p=value)
            assert settings.top_p == value

        # Invalid - too low
        with pytest.raises(ValueError):
            ProviderSettings(top_p=-0.1)

        # Invalid - too high
        with pytest.raises(ValueError):
            ProviderSettings(top_p=1.1)

    def test_top_k_validation(self):
        """Test top_k validation (>= 1)."""
        # Valid values
        for value in [1, 10, 100]:
            settings = ProviderSettings(top_k=value)
            assert settings.top_k == value

        # Invalid - zero
        with pytest.raises(ValueError):
            ProviderSettings(top_k=0)

        # Invalid - negative
        with pytest.raises(ValueError):
            ProviderSettings(top_k=-1)

    def test_max_tokens_validation(self):
        """Test max_tokens range validation (1 to 100000)."""
        # Valid values
        for value in [1, 1024, 100000]:
            settings = ProviderSettings(max_tokens=value)
            assert settings.max_tokens == value

        # Invalid - zero
        with pytest.raises(ValueError):
            ProviderSettings(max_tokens=0)

        # Invalid - too high
        with pytest.raises(ValueError):
            ProviderSettings(max_tokens=100001)

    def test_frequency_penalty_validation(self):
        """Test frequency_penalty range validation (-2.0 to 2.0)."""
        # Valid values
        for value in [-2.0, 0.0, 2.0]:
            settings = ProviderSettings(frequency_penalty=value)
            assert settings.frequency_penalty == value

        # Invalid - too low
        with pytest.raises(ValueError):
            ProviderSettings(frequency_penalty=-2.1)

        # Invalid - too high
        with pytest.raises(ValueError):
            ProviderSettings(frequency_penalty=2.1)

    def test_presence_penalty_validation(self):
        """Test presence_penalty range validation (-2.0 to 2.0)."""
        # Valid values
        for value in [-2.0, 0.0, 2.0]:
            settings = ProviderSettings(presence_penalty=value)
            assert settings.presence_penalty == value

        # Invalid - too low
        with pytest.raises(ValueError):
            ProviderSettings(presence_penalty=-2.1)

        # Invalid - too high
        with pytest.raises(ValueError):
            ProviderSettings(presence_penalty=2.1)

    def test_repeat_penalty_validation(self):
        """Test repeat_penalty validation (>= 0.0)."""
        # Valid values
        for value in [0.0, 1.0, 2.0]:
            settings = ProviderSettings(repeat_penalty=value)
            assert settings.repeat_penalty == value

        # Invalid - negative
        with pytest.raises(ValueError):
            ProviderSettings(repeat_penalty=-0.1)

    def test_timeout_validation(self):
        """Test timeout validation (>= 1)."""
        # Valid values
        for value in [1, 60, 300]:
            settings = ProviderSettings(timeout=value)
            assert settings.timeout == value

        # Invalid - zero
        with pytest.raises(ValueError):
            ProviderSettings(timeout=0)

        # Invalid - negative
        with pytest.raises(ValueError):
            ProviderSettings(timeout=-1)

    def test_merge_with_all_params(self):
        """Test merging two settings objects with all new parameters."""
        base = ProviderSettings(
            model="gpt-3.5",
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            max_tokens=2048,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repeat_penalty=1.0,
            timeout=60,
        )
        override = ProviderSettings(
            top_p=0.8,
            top_k=40,
            max_tokens=4096,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            repeat_penalty=1.2,
            timeout=120,
        )

        merged = base.merge_with(override)

        # From base (not overridden)
        assert merged.model == "gpt-3.5"
        assert merged.temperature == 0.5

        # From override
        assert merged.top_p == 0.8
        assert merged.top_k == 40
        assert merged.max_tokens == 4096
        assert merged.frequency_penalty == 0.5
        assert merged.presence_penalty == 0.3
        assert merged.repeat_penalty == 1.2
        assert merged.timeout == 120



class TestConfigSchema:
    """Tests for ConfigSchema model."""

    def test_create_with_defaults(self):
        """Test creating schema with defaults."""
        config = ConfigSchema()
        assert config.version == "1.0"
        assert config.defaults is not None
        assert config.providers == {}
        assert config.persona_configs == {}

    def test_create_with_values(self):
        """Test creating schema with explicit values."""
        config = ConfigSchema(
            version="1.1",
            defaults=ProviderSettings(model="gpt-4o"),
            providers={
                "openai": ProviderSettings(model="gpt-4o"),
            },
        )
        assert config.version == "1.1"
        assert config.defaults.model == "gpt-4o"
        assert "openai" in config.providers

    def test_nested_settings(self):
        """Test nested settings structures."""
        config = ConfigSchema(
            defaults=ProviderSettings(model="default"),
            persona_configs={
                "The Innovator": ProviderSettings(temperature=0.9),
            },
            council=CouncilSettings(max_rounds=10),
        )
        assert config.persona_configs["The Innovator"].temperature == 0.9
        assert config.council.max_rounds == 10


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_load_empty_config(self):
        """Test loading when no config files exist."""
        manager = ConfigManager()
        config = manager.load(skip_user=True, skip_project=True)
        assert config.version == "1.0"

    def test_load_from_yaml_file(self):
        """Test loading from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_config.yaml"
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump({
                    'version': '1.0',
                    'defaults': {
                        'model': 'test-model',
                        'temperature': 0.5,
                    }
                }, f)

            manager = ConfigManager()
            config = manager.load(
                skip_user=True,
                skip_project=True,
                config_path=str(filepath),
            )
            assert config.defaults.model == 'test-model'
            assert config.defaults.temperature == 0.5

    def test_merge_configs(self):
        """Test merging base and override configs."""
        manager = ConfigManager()

        base = ConfigSchema(
            defaults=ProviderSettings(model="base-model", temperature=0.5),
            providers={"provider1": ProviderSettings(model="p1")},
        )
        override = ConfigSchema(
            defaults=ProviderSettings(temperature=0.9),
            providers={"provider2": ProviderSettings(model="p2")},
        )

        merged = manager._merge_configs(base, override)

        assert merged.defaults.model == "base-model"
        assert merged.defaults.temperature == 0.9
        assert "provider1" in merged.providers
        assert "provider2" in merged.providers

    def test_resolve_with_cli_overrides(self):
        """Test resolving config with CLI overrides."""
        manager = ConfigManager()
        config = ConfigSchema(
            defaults=ProviderSettings(model="config-model", temperature=0.7)
        )

        resolved = manager.resolve(config, cli_overrides={
            "model": "cli-model",
            "temperature": 0.9,
        })

        assert resolved.defaults.model == "cli-model"
        assert resolved.defaults.temperature == 0.9
        assert resolved.sources.get("model") == "cli"

    def test_resolve_with_env_vars(self):
        """Test resolving config with environment variables."""
        manager = ConfigManager()
        config = ConfigSchema(
            defaults=ProviderSettings(model="config-model")
        )

        with patch.dict(os.environ, {"LLM_COUNCIL_MODEL": "env-model"}):
            resolved = manager.resolve(config)

        assert resolved.defaults.model == "env-model"
        assert resolved.sources.get("model") == "env:LLM_COUNCIL_MODEL"

    def test_get_provider_for_persona_default(self):
        """Test getting provider settings for persona with defaults."""
        manager = ConfigManager()
        resolved = ResolvedConfig(
            defaults=ProviderSettings(model="default-model", temperature=0.7),
            generation=GenerationSettings(),
            providers={},
            persona_configs={},
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )

        settings = manager.get_provider_for_persona("Unknown Persona", resolved)
        assert settings.model == "default-model"
        assert settings.temperature == 0.7

    def test_get_provider_for_persona_with_override(self):
        """Test getting provider settings for persona with overrides."""
        manager = ConfigManager()
        resolved = ResolvedConfig(
            defaults=ProviderSettings(model="default-model", temperature=0.7),
            generation=GenerationSettings(),
            providers={},
            persona_configs={
                "The Innovator": ProviderSettings(temperature=0.9),
            },
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )

        settings = manager.get_provider_for_persona("The Innovator", resolved)
        assert settings.model == "default-model"  # Inherited
        assert settings.temperature == 0.9  # Overridden

    def test_save_and_load_roundtrip(self):
        """Test saving and loading config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"

            config = ConfigSchema(
                defaults=ProviderSettings(model="test-model"),
                providers={"test": ProviderSettings(temperature=0.5)},
            )

            manager = ConfigManager()
            manager.save(config, path)

            assert path.exists()

            # Reload
            loaded = manager.load(
                skip_user=True,
                skip_project=True,
                config_path=str(path),
            )
            assert loaded.defaults.model == "test-model"
            assert "test" in loaded.providers


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        assert config.defaults.model == "openai/qwen/qwen3-coder-30b"
        assert config.defaults.api_base == "http://localhost:1234/v1"

    def test_load_config_no_files(self):
        """Test loading config when no files exist."""
        config = load_config(skip_user=True, skip_project=True)
        assert config is not None
        assert config.version == "1.0"


class TestConfigPaths:
    """Tests for configuration path functions."""

    def test_get_user_config_dir_windows(self):
        """Test user config dir on Windows."""
        with patch('os.name', 'nt'):
            with patch.dict(os.environ, {"APPDATA": "C:\\Users\\Test\\AppData\\Roaming"}):
                path = get_user_config_dir()
                assert "llm-council" in str(path)

    @pytest.mark.skipif(os.name == 'nt', reason="Unix path test not applicable on Windows")
    def test_get_user_config_dir_unix(self):
        """Test user config dir on Unix."""
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/home/test/.config"}):
            path = get_user_config_dir()
            assert "llm-council" in str(path)

    def test_get_project_config_path(self):
        """Test project config path."""
        path = get_project_config_path()
        assert path.name == ".llm-council.yaml"


class TestGenerationSettings:
    """Tests for GenerationSettings model."""

    def test_defaults(self):
        """Test default values."""
        settings = GenerationSettings()
        assert settings.temperature == 0.8
        assert settings.max_tokens == 2048

    def test_custom_values(self):
        """Test custom values."""
        settings = GenerationSettings(
            model="gpt-4o",
            temperature=0.5,
            prompt_template="Custom template",
        )
        assert settings.model == "gpt-4o"
        assert settings.prompt_template == "Custom template"


class TestCouncilSettings:
    """Tests for CouncilSettings model."""

    def test_defaults(self):
        """Test default values."""
        settings = CouncilSettings()
        assert settings.consensus_type == "majority"
        assert settings.max_rounds == 5
        assert settings.default_personas_count == 3

    def test_validation(self):
        """Test validation constraints."""
        # Valid
        settings = CouncilSettings(max_rounds=10)
        assert settings.max_rounds == 10

        # Invalid - too high
        with pytest.raises(ValueError):
            CouncilSettings(max_rounds=100)

        # Invalid - too low
        with pytest.raises(ValueError):
            CouncilSettings(max_rounds=0)


class TestPersistenceSettings:
    """Tests for PersistenceSettings model."""

    def test_defaults(self):
        """Test default values."""
        settings = PersistenceSettings()
        assert settings.enabled is True
        assert settings.retention_policy == "days_30"

    def test_custom_values(self):
        """Test custom values."""
        settings = PersistenceSettings(
            enabled=False,
            db_path="/custom/path.db",
            retention_policy="forever",
        )
        assert settings.enabled is False
        assert settings.db_path == "/custom/path.db"


class TestInheritanceChain:
    """Tests for configuration inheritance chain: defaults -> named provider -> persona overrides."""

    def _create_resolved_config(
        self,
        defaults: ProviderSettings,
        providers: dict = None,
        persona_configs: dict = None,
    ) -> ResolvedConfig:
        """Helper to create a ResolvedConfig for testing."""
        return ResolvedConfig(
            defaults=defaults,
            providers=providers or {},
            persona_configs=persona_configs or {},
            generation=GenerationSettings(),
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )

    def test_defaults_only(self):
        """Get provider for persona not in config, should return defaults."""
        defaults = ProviderSettings(
            model="default-model",
            temperature=0.7,
            max_tokens=1000,
            api_base="http://localhost:1234/v1",
        )
        resolved = self._create_resolved_config(defaults=defaults)
        manager = ConfigManager()

        settings = manager.get_provider_for_persona("unknown_persona", resolved)

        assert settings.model == "default-model"
        assert settings.temperature == 0.7
        assert settings.max_tokens == 1000
        assert settings.api_base == "http://localhost:1234/v1"

    def test_persona_overrides_defaults(self):
        """Persona config overrides default values."""
        defaults = ProviderSettings(
            model="default-model",
            temperature=0.7,
            max_tokens=1000,
        )
        persona_configs = {
            "creative": ProviderSettings(temperature=1.2, max_tokens=2000),
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings = manager.get_provider_for_persona("creative", resolved)

        assert settings.model == "default-model"  # Inherited from defaults
        assert settings.temperature == 1.2  # Overridden by persona
        assert settings.max_tokens == 2000  # Overridden by persona

    def test_named_provider_overrides_defaults(self):
        """Persona references named provider, provider merges over defaults."""
        defaults = ProviderSettings(
            model="default-model",
            temperature=0.7,
            max_tokens=1000,
            api_base="http://localhost:1234/v1",
        )
        providers = {
            "fast": ProviderSettings(temperature=0.3, max_tokens=500),
        }
        persona_configs = {
            "analyst": ProviderSettings(provider="fast"),
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings = manager.get_provider_for_persona("analyst", resolved)

        assert settings.model == "default-model"  # From defaults
        assert settings.api_base == "http://localhost:1234/v1"  # From defaults
        assert settings.temperature == 0.3  # From named provider "fast"
        assert settings.max_tokens == 500  # From named provider "fast"

    def test_full_chain_defaults_provider_persona(self):
        """Full chain: defaults + named provider + persona overrides."""
        defaults = ProviderSettings(
            model="default-model",
            temperature=0.7,
            max_tokens=1000,
            api_base="http://localhost:1234/v1",
            timeout=60,
        )
        providers = {
            "fast": ProviderSettings(temperature=0.3, max_tokens=500, timeout=30),
        }
        persona_configs = {
            "creative": ProviderSettings(provider="fast", temperature=1.0),
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings = manager.get_provider_for_persona("creative", resolved)

        assert settings.model == "default-model"  # From defaults
        assert settings.api_base == "http://localhost:1234/v1"  # From defaults
        assert settings.max_tokens == 500  # From named provider "fast"
        assert settings.timeout == 30  # From named provider "fast"
        assert settings.temperature == 1.0  # From persona override (overrides provider)

    def test_inheritance_temperature(self):
        """Temperature flows through inheritance correctly."""
        defaults = ProviderSettings(model="test-model", temperature=0.5)
        providers = {"hot": ProviderSettings(temperature=1.5)}
        persona_configs = {
            "inherit_default": ProviderSettings(),  # Should get 0.5
            "inherit_provider": ProviderSettings(provider="hot"),  # Should get 1.5
            "override_all": ProviderSettings(provider="hot", temperature=0.1),  # Should get 0.1
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings_default = manager.get_provider_for_persona("inherit_default", resolved)
        settings_provider = manager.get_provider_for_persona("inherit_provider", resolved)
        settings_override = manager.get_provider_for_persona("override_all", resolved)

        assert settings_default.temperature == 0.5
        assert settings_provider.temperature == 1.5
        assert settings_override.temperature == 0.1

    def test_inheritance_top_p(self):
        """top_p flows through inheritance correctly."""
        defaults = ProviderSettings(model="test-model", top_p=0.9)
        providers = {"precise": ProviderSettings(top_p=0.5)}
        persona_configs = {
            "inherit_default": ProviderSettings(),  # Should get 0.9
            "inherit_provider": ProviderSettings(provider="precise"),  # Should get 0.5
            "override_all": ProviderSettings(provider="precise", top_p=0.3),  # Should get 0.3
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings_default = manager.get_provider_for_persona("inherit_default", resolved)
        settings_provider = manager.get_provider_for_persona("inherit_provider", resolved)
        settings_override = manager.get_provider_for_persona("override_all", resolved)

        assert settings_default.top_p == 0.9
        assert settings_provider.top_p == 0.5
        assert settings_override.top_p == 0.3

    def test_inheritance_top_k(self):
        """top_k flows through inheritance correctly."""
        defaults = ProviderSettings(model="test-model", top_k=50)
        providers = {"narrow": ProviderSettings(top_k=10)}
        persona_configs = {
            "inherit_default": ProviderSettings(),  # Should get 50
            "inherit_provider": ProviderSettings(provider="narrow"),  # Should get 10
            "override_all": ProviderSettings(provider="narrow", top_k=5),  # Should get 5
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings_default = manager.get_provider_for_persona("inherit_default", resolved)
        settings_provider = manager.get_provider_for_persona("inherit_provider", resolved)
        settings_override = manager.get_provider_for_persona("override_all", resolved)

        assert settings_default.top_k == 50
        assert settings_provider.top_k == 10
        assert settings_override.top_k == 5

    def test_inheritance_max_tokens(self):
        """max_tokens flows through inheritance correctly."""
        defaults = ProviderSettings(model="test-model", max_tokens=2000)
        providers = {"concise": ProviderSettings(max_tokens=500)}
        persona_configs = {
            "inherit_default": ProviderSettings(),  # Should get 2000
            "inherit_provider": ProviderSettings(provider="concise"),  # Should get 500
            "override_all": ProviderSettings(provider="concise", max_tokens=100),  # Should get 100
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings_default = manager.get_provider_for_persona("inherit_default", resolved)
        settings_provider = manager.get_provider_for_persona("inherit_provider", resolved)
        settings_override = manager.get_provider_for_persona("override_all", resolved)

        assert settings_default.max_tokens == 2000
        assert settings_provider.max_tokens == 500
        assert settings_override.max_tokens == 100

    def test_inheritance_frequency_penalty(self):
        """frequency_penalty flows through inheritance correctly."""
        defaults = ProviderSettings(model="test-model", frequency_penalty=0.5)
        providers = {"diverse": ProviderSettings(frequency_penalty=1.5)}
        persona_configs = {
            "inherit_default": ProviderSettings(),  # Should get 0.5
            "inherit_provider": ProviderSettings(provider="diverse"),  # Should get 1.5
            "override_all": ProviderSettings(provider="diverse", frequency_penalty=-0.5),  # Should get -0.5
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings_default = manager.get_provider_for_persona("inherit_default", resolved)
        settings_provider = manager.get_provider_for_persona("inherit_provider", resolved)
        settings_override = manager.get_provider_for_persona("override_all", resolved)

        assert settings_default.frequency_penalty == 0.5
        assert settings_provider.frequency_penalty == 1.5
        assert settings_override.frequency_penalty == -0.5

    def test_inheritance_presence_penalty(self):
        """presence_penalty flows through inheritance correctly."""
        defaults = ProviderSettings(model="test-model", presence_penalty=0.3)
        providers = {"varied": ProviderSettings(presence_penalty=1.0)}
        persona_configs = {
            "inherit_default": ProviderSettings(),  # Should get 0.3
            "inherit_provider": ProviderSettings(provider="varied"),  # Should get 1.0
            "override_all": ProviderSettings(provider="varied", presence_penalty=-1.0),  # Should get -1.0
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings_default = manager.get_provider_for_persona("inherit_default", resolved)
        settings_provider = manager.get_provider_for_persona("inherit_provider", resolved)
        settings_override = manager.get_provider_for_persona("override_all", resolved)

        assert settings_default.presence_penalty == 0.3
        assert settings_provider.presence_penalty == 1.0
        assert settings_override.presence_penalty == -1.0

    def test_inheritance_repeat_penalty(self):
        """repeat_penalty flows through inheritance correctly."""
        defaults = ProviderSettings(model="test-model", repeat_penalty=1.0)
        providers = {"no_repeat": ProviderSettings(repeat_penalty=2.0)}
        persona_configs = {
            "inherit_default": ProviderSettings(),  # Should get 1.0
            "inherit_provider": ProviderSettings(provider="no_repeat"),  # Should get 2.0
            "override_all": ProviderSettings(provider="no_repeat", repeat_penalty=0.5),  # Should get 0.5
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings_default = manager.get_provider_for_persona("inherit_default", resolved)
        settings_provider = manager.get_provider_for_persona("inherit_provider", resolved)
        settings_override = manager.get_provider_for_persona("override_all", resolved)

        assert settings_default.repeat_penalty == 1.0
        assert settings_provider.repeat_penalty == 2.0
        assert settings_override.repeat_penalty == 0.5

    def test_inheritance_stop(self):
        """stop sequences flow through inheritance correctly."""
        defaults = ProviderSettings(model="test-model", stop=["<END>", "</s>"])
        providers = {"custom_stop": ProviderSettings(stop=["STOP", "DONE"])}
        persona_configs = {
            "inherit_default": ProviderSettings(),  # Should get ["<END>", "</s>"]
            "inherit_provider": ProviderSettings(provider="custom_stop"),  # Should get ["STOP", "DONE"]
            "override_all": ProviderSettings(provider="custom_stop", stop=["FINISHED"]),  # Should get ["FINISHED"]
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings_default = manager.get_provider_for_persona("inherit_default", resolved)
        settings_provider = manager.get_provider_for_persona("inherit_provider", resolved)
        settings_override = manager.get_provider_for_persona("override_all", resolved)

        assert settings_default.stop == ["<END>", "</s>"]
        assert settings_provider.stop == ["STOP", "DONE"]
        assert settings_override.stop == ["FINISHED"]

    def test_inheritance_seed(self):
        """seed flows through inheritance correctly."""
        defaults = ProviderSettings(model="test-model", seed=42)
        providers = {"fixed_seed": ProviderSettings(seed=12345)}
        persona_configs = {
            "inherit_default": ProviderSettings(),  # Should get 42
            "inherit_provider": ProviderSettings(provider="fixed_seed"),  # Should get 12345
            "override_all": ProviderSettings(provider="fixed_seed", seed=99999),  # Should get 99999
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings_default = manager.get_provider_for_persona("inherit_default", resolved)
        settings_provider = manager.get_provider_for_persona("inherit_provider", resolved)
        settings_override = manager.get_provider_for_persona("override_all", resolved)

        assert settings_default.seed == 42
        assert settings_provider.seed == 12345
        assert settings_override.seed == 99999

    def test_inheritance_timeout(self):
        """timeout flows through inheritance correctly."""
        defaults = ProviderSettings(model="test-model", timeout=60)
        providers = {"fast_timeout": ProviderSettings(timeout=10)}
        persona_configs = {
            "inherit_default": ProviderSettings(),  # Should get 60
            "inherit_provider": ProviderSettings(provider="fast_timeout"),  # Should get 10
            "override_all": ProviderSettings(provider="fast_timeout", timeout=120),  # Should get 120
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            providers=providers,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings_default = manager.get_provider_for_persona("inherit_default", resolved)
        settings_provider = manager.get_provider_for_persona("inherit_provider", resolved)
        settings_override = manager.get_provider_for_persona("override_all", resolved)

        assert settings_default.timeout == 60
        assert settings_provider.timeout == 10
        assert settings_override.timeout == 120

    def test_partial_override(self):
        """Only some params overridden, others preserved from defaults."""
        defaults = ProviderSettings(
            model="default-model",
            api_base="http://localhost:1234/v1",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            timeout=60,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        persona_configs = {
            "partial": ProviderSettings(
                temperature=1.0,  # Only override temperature
                max_tokens=2000,  # and max_tokens
            ),
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings = manager.get_provider_for_persona("partial", resolved)

        # Overridden values
        assert settings.temperature == 1.0
        assert settings.max_tokens == 2000
        # Preserved from defaults
        assert settings.model == "default-model"
        assert settings.api_base == "http://localhost:1234/v1"
        assert settings.top_p == 0.9
        assert settings.timeout == 60
        assert settings.frequency_penalty == 0.0
        assert settings.presence_penalty == 0.0

    def test_none_does_not_override(self):
        """Setting None in persona doesn't override default value."""
        defaults = ProviderSettings(
            model="default-model",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
        )
        # Create persona with explicit None values (simulating a sparse config)
        persona_configs = {
            "sparse": ProviderSettings(
                model=None,  # Explicit None should not override
                temperature=1.2,  # This should override
                max_tokens=None,  # Explicit None should not override
            ),
        }
        resolved = self._create_resolved_config(
            defaults=defaults,
            persona_configs=persona_configs,
        )
        manager = ConfigManager()

        settings = manager.get_provider_for_persona("sparse", resolved)

        # None values should not override - default values preserved
        assert settings.model == "default-model"
        assert settings.max_tokens == 1000
        assert settings.top_p == 0.9
        # Non-None value should override
        assert settings.temperature == 1.2


class TestValidateProviders:
    """Tests for provider validation functionality."""

    @pytest.mark.api
    def test_validate_providers_with_working_default(self):
        """Test validate_providers with working default provider."""
        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model="openai/qwen/qwen3-coder-30b",
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
        manager = ConfigManager()

        results = manager.validate_providers(resolved)

        assert "default" in results
        assert results["default"] is True

    def test_validate_providers_with_no_model(self):
        """Test validate_providers when no model is configured."""
        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model=None,  # No model
                api_base="http://localhost:1234/v1",
            ),
            providers={},
            persona_configs={},
            generation=GenerationSettings(),
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )
        manager = ConfigManager()

        results = manager.validate_providers(resolved)

        assert "default" in results
        assert results["default"] is False

    def test_validate_providers_with_invalid_endpoint(self):
        """Test validate_providers with unreachable endpoint."""
        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model="openai/test-model",
                api_base="http://localhost:59999/v1",  # Invalid port
                temperature=0.7,
                max_tokens=100,
                timeout=5,  # Short timeout
            ),
            providers={},
            persona_configs={},
            generation=GenerationSettings(),
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )
        manager = ConfigManager()

        results = manager.validate_providers(resolved)

        assert "default" in results
        assert results["default"] is False

    @pytest.mark.api
    def test_validate_providers_with_named_provider(self):
        """Test validate_providers validates named providers."""
        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model="openai/qwen/qwen3-coder-30b",
                api_base="http://localhost:1234/v1",
                temperature=0.7,
                max_tokens=100,
            ),
            providers={
                "lmstudio": ProviderSettings(
                    # Inherits model and api_base from defaults
                    temperature=0.5,
                ),
            },
            persona_configs={},
            generation=GenerationSettings(),
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )
        manager = ConfigManager()

        results = manager.validate_providers(resolved)

        assert "default" in results
        assert "lmstudio" in results
        # Both should be valid since they use same LM Studio endpoint
        assert results["default"] is True
        assert results["lmstudio"] is True

    def test_validate_providers_named_no_model(self):
        """Test validate_providers with named provider that has no model after merge."""
        resolved = ResolvedConfig(
            defaults=ProviderSettings(
                model=None,  # No model in defaults
                api_base="http://localhost:1234/v1",
            ),
            providers={
                "broken": ProviderSettings(
                    temperature=0.5,  # No model here either
                ),
            },
            persona_configs={},
            generation=GenerationSettings(),
            council=CouncilSettings(),
            persistence=PersistenceSettings(),
        )
        manager = ConfigManager()

        results = manager.validate_providers(resolved)

        assert "broken" in results
        assert results["broken"] is False


class TestConfigMergeEdgeCases:
    """Tests for edge cases in config merging."""

    def test_merge_persona_configs_override(self):
        """Test merging persona configs when both configs have same persona."""
        manager = ConfigManager()

        base = ConfigSchema(
            persona_configs={
                "The Innovator": ProviderSettings(temperature=0.9),
            },
        )
        override = ConfigSchema(
            persona_configs={
                "The Innovator": ProviderSettings(temperature=1.2, top_p=0.95),
            },
        )

        merged = manager._merge_configs(base, override)

        # Should merge the persona configs
        assert merged.persona_configs["The Innovator"].temperature == 1.2
        assert merged.persona_configs["The Innovator"].top_p == 0.95

    def test_merge_providers_override(self):
        """Test merging provider configs when both configs have same provider."""
        manager = ConfigManager()

        base = ConfigSchema(
            providers={
                "fast": ProviderSettings(temperature=0.3, max_tokens=500),
            },
        )
        override = ConfigSchema(
            providers={
                "fast": ProviderSettings(temperature=0.2),  # Only temperature
            },
        )

        merged = manager._merge_configs(base, override)

        # Temperature overridden, max_tokens preserved
        assert merged.providers["fast"].temperature == 0.2
        assert merged.providers["fast"].max_tokens == 500

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file produces warning."""
        import warnings

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "invalid.yaml"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("invalid: yaml: content: [unclosed")

            manager = ConfigManager()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = manager._load_yaml(filepath)

                assert result is None
                assert len(w) == 1
                assert "Failed to load config" in str(w[0].message)

    def test_load_empty_yaml(self):
        """Test loading empty YAML file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty.yaml"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("")  # Empty file

            manager = ConfigManager()
            result = manager._load_yaml(filepath)

            assert result is None


class TestEnvVarConversions:
    """Tests for environment variable type conversions in resolve."""

    def test_env_var_temperature_conversion(self):
        """Test temperature from env var is converted to float."""
        manager = ConfigManager()
        config = ConfigSchema(
            defaults=ProviderSettings(model="test-model", temperature=0.5)
        )

        with patch.dict(os.environ, {"LLM_COUNCIL_TEMPERATURE": "0.9"}):
            resolved = manager.resolve(config)

        assert resolved.defaults.temperature == 0.9
        assert isinstance(resolved.defaults.temperature, float)

    def test_env_var_max_tokens_conversion(self):
        """Test max_tokens from env var is converted to int."""
        manager = ConfigManager()
        config = ConfigSchema(
            defaults=ProviderSettings(model="test-model", max_tokens=1000)
        )

        with patch.dict(os.environ, {"LLM_COUNCIL_MAX_TOKENS": "2048"}):
            resolved = manager.resolve(config)

        assert resolved.defaults.max_tokens == 2048
        assert isinstance(resolved.defaults.max_tokens, int)

    def test_env_var_timeout_conversion(self):
        """Test timeout from env var is converted to int."""
        manager = ConfigManager()
        config = ConfigSchema(
            defaults=ProviderSettings(model="test-model", timeout=60)
        )

        with patch.dict(os.environ, {"LLM_COUNCIL_TIMEOUT": "120"}):
            resolved = manager.resolve(config)

        assert resolved.defaults.timeout == 120
        assert isinstance(resolved.defaults.timeout, int)

    def test_env_var_api_base_override(self):
        """Test api_base from env var."""
        manager = ConfigManager()
        config = ConfigSchema(
            defaults=ProviderSettings(
                model="test-model",
                api_base="http://localhost:1234/v1"
            )
        )

        with patch.dict(os.environ, {"LLM_COUNCIL_API_BASE": "http://other:5678/v1"}):
            resolved = manager.resolve(config)

        assert resolved.defaults.api_base == "http://other:5678/v1"
        assert resolved.sources["api_base"] == "env:LLM_COUNCIL_API_BASE"

    def test_env_var_api_key_override(self):
        """Test api_key from env var."""
        manager = ConfigManager()
        config = ConfigSchema(
            defaults=ProviderSettings(model="test-model", api_key="default-key")
        )

        with patch.dict(os.environ, {"LLM_COUNCIL_API_KEY": "env-key"}):
            resolved = manager.resolve(config)

        assert resolved.defaults.api_key == "env-key"


class TestProviderEnvVarResolution:
    """Tests for env var resolution in providers and persona configs."""

    def test_resolve_env_vars_in_providers(self):
        """Test env vars are resolved in named providers."""
        manager = ConfigManager()
        config = ConfigSchema(
            defaults=ProviderSettings(model="default"),
            providers={
                "test_provider": ProviderSettings(
                    api_key="${TEST_API_KEY}",
                    api_base="${TEST_API_BASE}",
                ),
            },
        )

        with patch.dict(os.environ, {
            "TEST_API_KEY": "resolved-key",
            "TEST_API_BASE": "http://resolved:1234/v1",
        }):
            resolved = manager.resolve(config)

        assert resolved.providers["test_provider"].api_key == "resolved-key"
        assert resolved.providers["test_provider"].api_base == "http://resolved:1234/v1"

    def test_resolve_env_vars_in_persona_configs(self):
        """Test env vars are resolved in persona configs."""
        manager = ConfigManager()
        config = ConfigSchema(
            defaults=ProviderSettings(model="default"),
            persona_configs={
                "The Innovator": ProviderSettings(
                    api_key="${PERSONA_API_KEY}",
                ),
            },
        )

        with patch.dict(os.environ, {"PERSONA_API_KEY": "persona-key"}):
            resolved = manager.resolve(config)

        assert resolved.persona_configs["The Innovator"].api_key == "persona-key"
