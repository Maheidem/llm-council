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
