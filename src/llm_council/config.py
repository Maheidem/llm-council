"""Configuration management for LLM Council.

Supports:
- User config (~/.llm-council/config.yaml or %APPDATA%/llm-council/config.yaml)
- Project config (./.llm-council.yaml)
- Environment variable overrides
- Per-persona provider configuration
"""

import os
import re
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# Environment variable prefix
ENV_PREFIX = "LLM_COUNCIL_"

# Environment variable pattern for resolution: ${VAR_NAME}
ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def get_user_config_dir() -> Path:
    """Get platform-appropriate user config directory."""
    if os.name == 'nt':  # Windows
        base = os.environ.get('APPDATA', os.path.expanduser('~'))
        return Path(base) / 'llm-council'
    else:  # Unix/Linux/Mac
        xdg_config = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
        return Path(xdg_config) / 'llm-council'


def get_user_config_path() -> Path:
    """Get path to user config file."""
    return get_user_config_dir() / 'config.yaml'


def get_project_config_path() -> Path:
    """Get path to project config file."""
    return Path.cwd() / '.llm-council.yaml'


def resolve_env_vars(value: Any) -> Any:
    """Resolve ${ENV_VAR} references in strings."""
    if isinstance(value, str):
        def replace_env(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                warnings.warn(f"Environment variable {var_name} not found")
                return match.group(0)  # Keep original
            return env_value
        return ENV_VAR_PATTERN.sub(replace_env, value)
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(v) for v in value]
    return value


class ProviderSettings(BaseModel):
    """Settings for an LLM provider."""
    model: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[str] = None  # Reference to a named provider for inheritance
    # Sampling
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)
    max_tokens: Optional[int] = Field(None, ge=1, le=100000)
    # Repetition control
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    repeat_penalty: Optional[float] = Field(None, ge=0.0)  # LM Studio extension
    # Control
    stop: Optional[list[str]] = None
    seed: Optional[int] = None
    timeout: Optional[int] = Field(None, ge=1)

    @field_validator('api_key')
    @classmethod
    def warn_plaintext_key(cls, v):
        """Warn if API key appears to be plaintext (not env var reference)."""
        if v and not v.startswith('${') and len(v) > 20:
            warnings.warn(
                "API key appears to be stored in plaintext. "
                "Consider using ${ENV_VAR} syntax for security.",
                UserWarning
            )
        return v

    def merge_with(self, other: 'ProviderSettings') -> 'ProviderSettings':
        """Merge this config with another, other takes precedence for non-None values."""
        data = self.model_dump()
        other_data = other.model_dump()
        for key, value in other_data.items():
            if value is not None:
                data[key] = value
        return ProviderSettings(**data)

    def resolve_env_vars(self) -> 'ProviderSettings':
        """Resolve environment variables in all string fields."""
        data = self.model_dump()
        resolved = resolve_env_vars(data)
        return ProviderSettings(**resolved)


class GenerationSettings(BaseModel):
    """Settings for persona generation."""
    model: Optional[str] = None
    provider: Optional[str] = None  # Named provider to use
    temperature: float = 0.8
    max_tokens: int = 2048
    prompt_template: Optional[str] = None


class CouncilSettings(BaseModel):
    """Settings for council sessions."""
    consensus_type: str = "majority"
    max_rounds: int = Field(5, ge=1, le=20)
    stalemate_threshold: int = Field(2, ge=1)
    default_personas_count: int = Field(3, ge=2, le=10)
    auto_personas: bool = False


class PersistenceSettings(BaseModel):
    """Settings for session persistence."""
    enabled: bool = True
    db_path: Optional[str] = None
    retention_policy: str = "days_30"


class ConfigSchema(BaseModel):
    """Root configuration schema."""
    version: str = "1.0"
    defaults: ProviderSettings = Field(default_factory=ProviderSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    providers: Dict[str, ProviderSettings] = Field(default_factory=dict)
    persona_configs: Dict[str, ProviderSettings] = Field(default_factory=dict)
    council: CouncilSettings = Field(default_factory=CouncilSettings)
    persistence: PersistenceSettings = Field(default_factory=PersistenceSettings)

    @model_validator(mode='after')
    def validate_provider_references(self):
        """Validate that persona_configs reference valid providers or have inline config."""
        # Just validate structure, don't require provider references to exist
        return self


@dataclass
class ResolvedConfig:
    """Fully resolved configuration with all merges applied."""
    defaults: ProviderSettings
    generation: GenerationSettings
    providers: Dict[str, ProviderSettings]
    persona_configs: Dict[str, ProviderSettings]
    council: CouncilSettings
    persistence: PersistenceSettings

    # Metadata about sources
    sources: Dict[str, str] = field(default_factory=dict)


class ConfigManager:
    """Manages configuration loading, merging, and resolution."""

    def __init__(self):
        self._user_config: Optional[ConfigSchema] = None
        self._project_config: Optional[ConfigSchema] = None
        self._loaded = False

    def load(
        self,
        skip_user: bool = False,
        skip_project: bool = False,
        config_path: Optional[str] = None,
    ) -> ConfigSchema:
        """Load configuration from all sources.

        Args:
            skip_user: Skip loading user config
            skip_project: Skip loading project config
            config_path: Explicit config file path (overrides auto-discovery)
        """
        # Start with defaults
        merged = ConfigSchema()

        # Load user config
        if not skip_user and config_path is None:
            user_path = get_user_config_path()
            if user_path.exists():
                self._user_config = self._load_yaml(user_path)
                if self._user_config:
                    merged = self._merge_configs(merged, self._user_config)

        # Load project config
        if not skip_project and config_path is None:
            project_path = get_project_config_path()
            if project_path.exists():
                self._project_config = self._load_yaml(project_path)
                if self._project_config:
                    merged = self._merge_configs(merged, self._project_config)

        # Load explicit config path
        if config_path:
            explicit_config = self._load_yaml(Path(config_path))
            if explicit_config:
                merged = self._merge_configs(merged, explicit_config)

        self._loaded = True
        return merged

    def _load_yaml(self, path: Path) -> Optional[ConfigSchema]:
        """Load and validate a YAML config file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if data is None:
                return None
            return ConfigSchema(**data)
        except Exception as e:
            warnings.warn(f"Failed to load config from {path}: {e}")
            return None

    def _merge_configs(self, base: ConfigSchema, override: ConfigSchema) -> ConfigSchema:
        """Merge two configs, override takes precedence."""
        # Merge defaults
        merged_defaults = base.defaults.merge_with(override.defaults)

        # Merge providers (override adds/replaces)
        merged_providers = {**base.providers}
        for name, settings in override.providers.items():
            if name in merged_providers:
                merged_providers[name] = merged_providers[name].merge_with(settings)
            else:
                merged_providers[name] = settings

        # Merge persona_configs (override adds/replaces)
        merged_persona_configs = {**base.persona_configs}
        for name, settings in override.persona_configs.items():
            if name in merged_persona_configs:
                merged_persona_configs[name] = merged_persona_configs[name].merge_with(settings)
            else:
                merged_persona_configs[name] = settings

        # Generation settings (merge)
        gen_data = base.generation.model_dump()
        override_gen = override.generation.model_dump()
        for k, v in override_gen.items():
            if v is not None:
                gen_data[k] = v
        merged_generation = GenerationSettings(**gen_data)

        # Council settings (merge)
        council_data = base.council.model_dump()
        council_data.update({k: v for k, v in override.council.model_dump().items() if v is not None})
        merged_council = CouncilSettings(**council_data)

        # Persistence settings (merge)
        persist_data = base.persistence.model_dump()
        persist_data.update({k: v for k, v in override.persistence.model_dump().items() if v is not None})
        merged_persistence = PersistenceSettings(**persist_data)

        return ConfigSchema(
            version=override.version or base.version,
            defaults=merged_defaults,
            generation=merged_generation,
            providers=merged_providers,
            persona_configs=merged_persona_configs,
            council=merged_council,
            persistence=merged_persistence,
        )

    def resolve(
        self,
        config: ConfigSchema,
        cli_overrides: Optional[Dict[str, Any]] = None,
    ) -> ResolvedConfig:
        """Resolve final config with CLI overrides and env vars.

        Priority (highest to lowest):
        1. CLI arguments
        2. Environment variables
        3. Config file values
        """
        sources: Dict[str, str] = {}

        # Start with config defaults
        defaults_data = config.defaults.model_dump()

        # Apply environment variable overrides
        env_mappings = {
            'LLM_COUNCIL_MODEL': 'model',
            'LLM_COUNCIL_API_BASE': 'api_base',
            'LLM_COUNCIL_API_KEY': 'api_key',
            'LLM_COUNCIL_TEMPERATURE': 'temperature',
            'LLM_COUNCIL_MAX_TOKENS': 'max_tokens',
            'LLM_COUNCIL_TIMEOUT': 'timeout',
        }

        for env_var, config_key in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Type conversion
                if config_key in ('temperature',):
                    env_value = float(env_value)
                elif config_key in ('max_tokens', 'timeout', 'top_k', 'context_size'):
                    env_value = int(env_value)
                defaults_data[config_key] = env_value
                sources[config_key] = f'env:{env_var}'

        # Apply CLI overrides
        if cli_overrides:
            for key, value in cli_overrides.items():
                if value is not None and key in defaults_data:
                    defaults_data[key] = value
                    sources[key] = 'cli'

        # Resolve env vars in all settings
        resolved_defaults = ProviderSettings(**defaults_data).resolve_env_vars()

        # Resolve env vars in providers
        resolved_providers = {}
        for name, settings in config.providers.items():
            resolved_providers[name] = settings.resolve_env_vars()

        # Resolve env vars in persona configs
        resolved_persona_configs = {}
        for name, settings in config.persona_configs.items():
            resolved_persona_configs[name] = settings.resolve_env_vars()

        return ResolvedConfig(
            defaults=resolved_defaults,
            generation=config.generation,
            providers=resolved_providers,
            persona_configs=resolved_persona_configs,
            council=config.council,
            persistence=config.persistence,
            sources=sources,
        )

    def save(self, config: ConfigSchema, path: Optional[Path] = None):
        """Save configuration to file."""
        if path is None:
            path = get_user_config_path()

        path.parent.mkdir(parents=True, exist_ok=True)

        data = config.model_dump(exclude_none=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_provider_for_persona(
        self,
        persona_name: str,
        resolved: ResolvedConfig,
    ) -> ProviderSettings:
        """Get resolved provider settings for a specific persona.

        Uses merge inheritance: persona settings override defaults.
        """
        # Start with defaults
        settings = resolved.defaults

        # Check for persona-specific config
        if persona_name in resolved.persona_configs:
            persona_settings = resolved.persona_configs[persona_name]

            # Check if it references a named provider
            provider_ref = getattr(persona_settings, 'provider', None)
            if provider_ref and provider_ref in resolved.providers:
                # Merge: defaults -> named provider -> persona overrides
                settings = settings.merge_with(resolved.providers[provider_ref])

            # Apply persona-specific overrides
            settings = settings.merge_with(persona_settings)

        return settings

    def validate_providers(self, resolved: ResolvedConfig) -> Dict[str, bool]:
        """Validate all configured providers (eager validation).

        Returns dict of provider_name -> is_valid.
        """
        from .providers import create_provider

        results = {}

        # Validate default provider
        try:
            defaults = resolved.defaults
            if defaults.model:
                provider = create_provider(
                    model=defaults.model,
                    api_base=defaults.api_base,
                    api_key=defaults.api_key,
                    temperature=defaults.temperature or 0.7,
                    max_tokens=defaults.max_tokens or 1024,
                )
                results['default'] = provider.test_connection()
            else:
                results['default'] = False
        except Exception as e:
            results['default'] = False
            warnings.warn(f"Default provider validation failed: {e}")

        # Validate named providers
        for name, settings in resolved.providers.items():
            try:
                merged = resolved.defaults.merge_with(settings)
                if merged.model:
                    provider = create_provider(
                        model=merged.model,
                        api_base=merged.api_base,
                        api_key=merged.api_key,
                        temperature=merged.temperature or 0.7,
                        max_tokens=merged.max_tokens or 1024,
                    )
                    results[name] = provider.test_connection()
                else:
                    results[name] = False
            except Exception as e:
                results[name] = False
                warnings.warn(f"Provider '{name}' validation failed: {e}")

        return results


# Convenience functions
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get or create the global config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(
    skip_user: bool = False,
    skip_project: bool = False,
    config_path: Optional[str] = None,
) -> ConfigSchema:
    """Load configuration from all sources."""
    return get_config_manager().load(
        skip_user=skip_user,
        skip_project=skip_project,
        config_path=config_path,
    )


def get_default_config() -> ConfigSchema:
    """Get default configuration (no files loaded)."""
    return ConfigSchema(
        defaults=ProviderSettings(
            model="openai/qwen/qwen3-coder-30b",
            api_base="http://localhost:1234/v1",
            api_key="lm-studio",
            temperature=0.7,
            max_tokens=1024,
            timeout=120,
        )
    )


def save_config(config: ConfigSchema, path: Optional[Path] = None):
    """Save configuration to file."""
    get_config_manager().save(config, path)
