"""MCP Server for LLM Council - Exposes council discussions as MCP tools.

Provides:
- Council discussion orchestration
- Configuration management (get/set/init/validate)
- Provider management
"""

import json
import os
from pathlib import Path
from typing import Optional, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .models import ConsensusType
from .providers import create_provider, ProviderRegistry
from .personas import PersonaManager
from .council import CouncilEngine
from .config import (
    ConfigManager,
    ConfigSchema,
    ProviderSettings,
    ResolvedConfig,
    get_config_manager,
    get_default_config,
    get_user_config_path,
    get_project_config_path,
    load_config,
    save_config,
)


# Create server instance
server = Server("llm-council")


@server.list_tools()
async def list_tools():
    """List available tools."""
    return [
        # Main council discussion tool
        Tool(
            name="council_discuss",
            description="Run a council discussion with multiple AI personas to reach consensus on a topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to discuss"
                    },
                    "objective": {
                        "type": "string",
                        "description": "The goal or decision to reach"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context for the discussion (optional)"
                    },
                    "personas": {
                        "type": "integer",
                        "description": "Number of personas (default: 3)",
                        "default": 3
                    },
                    "personas_file": {
                        "type": "string",
                        "description": "Path to YAML/JSON file with persona definitions. Overrides 'personas' count if provided."
                    },
                    "max_rounds": {
                        "type": "integer",
                        "description": "Maximum discussion rounds (default: 3)",
                        "default": 3
                    },
                    "consensus_type": {
                        "type": "string",
                        "enum": ["unanimous", "supermajority", "majority", "plurality"],
                        "description": "Type of consensus required (default: majority)",
                        "default": "majority"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use (overrides config)"
                    },
                    "api_base": {
                        "type": "string",
                        "description": "API base URL (overrides config)"
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key if required (overrides config)"
                    }
                },
                "required": ["topic", "objective"]
            }
        ),
        # Configuration management tools
        Tool(
            name="config_get",
            description="Get current LLM Council configuration values. Shows default provider settings, council settings, and any per-persona configurations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Specific config key to get (e.g., 'defaults.model', 'council.max_rounds'). If omitted, returns full config."
                    },
                    "resolved": {
                        "type": "boolean",
                        "description": "If true, show resolved values with env vars expanded. Default: false.",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="config_set",
            description="Set LLM Council configuration values. Changes are saved to user config file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Config key to set (e.g., 'defaults.model', 'defaults.api_base', 'council.max_rounds')"
                    },
                    "value": {
                        "type": ["string", "number", "boolean"],
                        "description": "Value to set"
                    }
                },
                "required": ["key", "value"]
            }
        ),
        Tool(
            name="config_init",
            description="Initialize LLM Council configuration with a 3-step guided setup. Use this for first-time setup or to reconfigure. Returns step-by-step form structures for: (1) Provider & model selection, (2) API key configuration with security guidance, (3) Connection validation before saving. Call without parameters to start the wizard, or with all parameters for quick setup.",
            inputSchema={
                "type": "object",
                "properties": {
                    "preset": {
                        "type": "string",
                        "enum": ["local", "openai", "anthropic", "custom"],
                        "description": "Configuration preset to use. 'local' for LM Studio/Ollama, 'openai' for OpenAI API, 'anthropic' for Claude API, 'custom' for manual setup."
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name/identifier to use. For OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo. For Anthropic: anthropic/claude-sonnet-4-20250514, anthropic/claude-opus-4-20250514, anthropic/claude-3-5-haiku-20241022."
                    },
                    "api_base": {
                        "type": "string",
                        "description": "API base URL (required for local/custom). LM Studio: http://localhost:1234/v1, Ollama: http://localhost:11434/v1"
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key (use ${ENV_VAR} syntax for security, e.g., ${OPENAI_API_KEY})"
                    },
                    "skip_validation": {
                        "type": "boolean",
                        "description": "Skip connection validation before saving (not recommended). Default: false.",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="config_validate",
            description="Validate LLM Council configuration by testing provider connections. Returns validation status for each configured provider.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Specific provider to validate. If omitted, validates all providers."
                    }
                }
            }
        ),
        Tool(
            name="providers_list",
            description="List all configured LLM providers and their status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "show_details": {
                        "type": "boolean",
                        "description": "Show full provider configuration details. Default: false.",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="personas_generate",
            description="Generate custom personas for a specific topic. Creates diverse perspectives relevant to the discussion subject.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to generate personas for"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of personas to generate (default: 3)",
                        "default": 3,
                        "minimum": 2,
                        "maximum": 10
                    },
                    "save_to": {
                        "type": "string",
                        "description": "Optional file path to save generated personas (YAML or JSON)"
                    }
                },
                "required": ["topic"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls."""
    handlers = {
        "council_discuss": run_council_discussion,
        "config_get": handle_config_get,
        "config_set": handle_config_set,
        "config_init": handle_config_init,
        "config_validate": handle_config_validate,
        "providers_list": handle_providers_list,
        "personas_generate": handle_personas_generate,
    }

    handler = handlers.get(name)
    if handler:
        return await handler(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_council_discussion(args: dict) -> list[TextContent]:
    """Run a council discussion and return results."""
    topic = args.get("topic")
    objective = args.get("objective")
    context = args.get("context")
    num_personas = args.get("personas", 3)
    personas_file = args.get("personas_file")
    max_rounds = args.get("max_rounds", 3)
    consensus_type_str = args.get("consensus_type", "majority")

    # Load config and apply overrides
    config = load_config()
    manager = get_config_manager()
    resolved = manager.resolve(config, cli_overrides={
        "model": args.get("model"),
        "api_base": args.get("api_base"),
        "api_key": args.get("api_key"),
    })

    # Get final provider settings
    defaults = resolved.defaults
    model = defaults.model or "openai/qwen/qwen3-coder-30b"
    api_base = defaults.api_base or "http://localhost:1234/v1"
    api_key = defaults.api_key

    try:
        # Create provider
        provider = create_provider(
            provider_type="litellm",
            model=model,
            api_base=api_base,
            api_key=api_key,
        )

        # Get personas - file takes precedence over count
        persona_manager = PersonaManager(provider=provider)
        if personas_file:
            try:
                personas = persona_manager.load_personas(personas_file)
            except FileNotFoundError:
                return [TextContent(type="text", text=f"Error: Personas file not found: {personas_file}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error loading personas file: {str(e)}")]
        else:
            personas = persona_manager.get_default_personas(num_personas)

        # Create engine
        engine = CouncilEngine(
            provider=provider,
            consensus_type=ConsensusType(consensus_type_str),
            max_rounds=max_rounds,
        )

        # Run session
        session = engine.run_session(
            topic=topic,
            objective=objective,
            personas=personas,
            initial_context=context,
        )

        # Format result
        result = {
            "topic": session.topic,
            "objective": session.objective,
            "consensus_reached": session.consensus_reached,
            "final_consensus": session.final_consensus,
            "rounds_completed": len(session.rounds),
            "personas": [p.name for p in personas],
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_config_get(args: dict) -> list[TextContent]:
    """Get configuration values."""
    key = args.get("key")
    show_resolved = args.get("resolved", False)

    try:
        config = load_config()

        if show_resolved:
            manager = get_config_manager()
            resolved = manager.resolve(config)
            data = {
                "defaults": resolved.defaults.model_dump(exclude_none=True),
                "generation": resolved.generation.model_dump(exclude_none=True),
                "providers": {k: v.model_dump(exclude_none=True) for k, v in resolved.providers.items()},
                "persona_configs": {k: v.model_dump(exclude_none=True) for k, v in resolved.persona_configs.items()},
                "council": resolved.council.model_dump(exclude_none=True),
                "persistence": resolved.persistence.model_dump(exclude_none=True),
                "sources": resolved.sources,
            }
        else:
            data = config.model_dump(exclude_none=True)

        # Navigate to specific key if provided
        if key:
            parts = key.split(".")
            result = data
            for part in parts:
                if isinstance(result, dict) and part in result:
                    result = result[part]
                else:
                    return [TextContent(type="text", text=f"Key not found: {key}")]
            data = {key: result}

        return [TextContent(type="text", text=json.dumps(data, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting config: {str(e)}")]


async def handle_config_set(args: dict) -> list[TextContent]:
    """Set a configuration value."""
    key = args.get("key")
    value = args.get("value")

    if not key:
        return [TextContent(type="text", text="Error: 'key' is required")]

    try:
        # Load existing config or create default
        config_path = get_user_config_path()
        if config_path.exists():
            config = load_config(skip_project=True)
        else:
            config = get_default_config()

        # Parse the key path and set value
        data = config.model_dump()
        parts = key.split(".")
        target = data
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Set the final value
        target[parts[-1]] = value

        # Save back
        new_config = ConfigSchema(**data)
        save_config(new_config, config_path)

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "message": f"Set {key} = {value}",
            "config_path": str(config_path),
        }, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error setting config: {str(e)}")]


async def handle_config_init(args: dict) -> list[TextContent]:
    """Initialize configuration - 3-step onboarding flow.

    Steps:
    1. Provider & Model Selection - Choose provider preset and model
    2. API Key Entry - Configure API key with guidance (cloud providers only)
    3. Connection Validation - Test connection before saving

    Backward compatible: config_init(preset="local") with all required params
    still works for quick setup.
    """
    preset = args.get("preset")
    model = args.get("model")
    api_base = args.get("api_base")
    api_key = args.get("api_key")
    skip_validation = args.get("skip_validation", False)

    # Model options per provider
    MODEL_OPTIONS = {
        "openai": [
            {"value": "gpt-4o", "label": "GPT-4o", "description": "Most capable model, best for complex discussions", "recommended": True},
            {"value": "gpt-4o-mini", "label": "GPT-4o Mini", "description": "Faster and cheaper, good for most use cases"},
            {"value": "gpt-4-turbo", "label": "GPT-4 Turbo", "description": "Previous generation, still very capable"},
            {"value": "custom", "label": "Other model", "description": "Enter a custom model name"},
        ],
        "anthropic": [
            {"value": "anthropic/claude-sonnet-4-20250514", "label": "Claude Sonnet 4", "description": "Latest balanced model, best for most tasks", "recommended": True},
            {"value": "anthropic/claude-opus-4-20250514", "label": "Claude Opus 4", "description": "Most capable, best for complex discussions"},
            {"value": "anthropic/claude-3-5-haiku-20241022", "label": "Claude 3.5 Haiku", "description": "Fastest and most affordable"},
            {"value": "custom", "label": "Other model", "description": "Enter a custom model name"},
        ],
    }

    # API key URLs per provider
    API_KEY_URLS = {
        "openai": "https://platform.openai.com/api-keys",
        "anthropic": "https://console.anthropic.com/settings/keys",
    }

    # ========================================================================
    # STEP 1: Provider Selection (no preset provided)
    # ========================================================================
    if not preset:
        return [TextContent(type="text", text=json.dumps({
            "step": "provider_select",
            "step_number": 1,
            "total_steps": 3,
            "message": "Welcome to LLM Council setup! Let's configure your LLM provider.",
            "questions": [
                {
                    "id": "preset",
                    "type": "select",
                    "question": "Which LLM provider would you like to use?",
                    "options": [
                        {
                            "value": "local",
                            "label": "Local Server",
                            "description": "LM Studio, Ollama, or other local LLM server",
                            "requires_api_key": False,
                        },
                        {
                            "value": "openai",
                            "label": "OpenAI",
                            "description": "GPT-4o, GPT-4o-mini, and other OpenAI models",
                            "requires_api_key": True,
                            "api_key_url": API_KEY_URLS["openai"],
                        },
                        {
                            "value": "anthropic",
                            "label": "Anthropic Claude",
                            "description": "Claude Opus, Sonnet, and Haiku models",
                            "requires_api_key": True,
                            "api_key_url": API_KEY_URLS["anthropic"],
                        },
                        {
                            "value": "custom",
                            "label": "Custom Endpoint",
                            "description": "Any OpenAI-compatible API endpoint",
                            "requires_api_key": "optional",
                        },
                    ],
                }
            ],
            "next_action": "Call config_init(preset='<selected>') to continue",
        }, indent=2))]

    # Validate preset
    valid_presets = ["local", "openai", "anthropic", "custom"]
    if preset not in valid_presets:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Unknown preset: {preset}",
            "valid_presets": valid_presets,
            "troubleshooting": [
                "Check the preset value is one of: local, openai, anthropic, custom",
                "Call config_init() without parameters to see all options",
            ],
        }, indent=2))]

    # ========================================================================
    # STEP 1b: Model Selection (preset provided, but no model)
    # ========================================================================

    # For local/custom without model, ask for model and api_base
    if preset == "local" and not model:
        return [TextContent(type="text", text=json.dumps({
            "step": "model_select",
            "step_number": 1,
            "total_steps": 3,
            "preset": preset,
            "message": "Configure your local LLM server.",
            "questions": [
                {
                    "id": "api_base",
                    "type": "text",
                    "question": "What is your local server URL?",
                    "default": "http://localhost:1234/v1",
                    "hint": "LM Studio default: http://localhost:1234/v1, Ollama: http://localhost:11434/v1",
                },
                {
                    "id": "model",
                    "type": "text",
                    "question": "What model are you running?",
                    "default": "openai/qwen/qwen3-coder-30b",
                    "hint": "Use 'openai/' prefix for OpenAI-compatible endpoints",
                },
            ],
            "next_action": "Call config_init(preset='local', api_base='...', model='...') to continue",
        }, indent=2))]

    if preset == "custom" and (not model or not api_base):
        return [TextContent(type="text", text=json.dumps({
            "step": "model_select",
            "step_number": 1,
            "total_steps": 3,
            "preset": preset,
            "message": "Configure your custom LLM endpoint.",
            "questions": [
                {
                    "id": "api_base",
                    "type": "text",
                    "question": "What is your API endpoint URL?",
                    "required": True,
                    "hint": "e.g., https://api.example.com/v1",
                },
                {
                    "id": "model",
                    "type": "text",
                    "question": "What is the model identifier?",
                    "required": True,
                    "hint": "e.g., openai/gpt-4o or custom-model-name",
                },
            ],
            "next_action": "Call config_init(preset='custom', api_base='...', model='...') to continue",
        }, indent=2))]

    # For cloud providers without model, show model selection
    if preset in ["openai", "anthropic"] and not model:
        return [TextContent(type="text", text=json.dumps({
            "step": "model_select",
            "step_number": 1,
            "total_steps": 3,
            "preset": preset,
            "message": f"Select which {preset.title()} model you'd like to use.",
            "questions": [
                {
                    "id": "model",
                    "type": "select",
                    "question": "Select model:",
                    "options": MODEL_OPTIONS[preset],
                    "allow_custom": True,
                    "custom_hint": "Enter model name (e.g., gpt-4o-2024-11-20)" if preset == "openai" else "Enter model name (e.g., anthropic/claude-3-5-sonnet-20241022)",
                }
            ],
            "next_action": f"Call config_init(preset='{preset}', model='<selected>') to continue",
        }, indent=2))]

    # Handle "custom" model selection - user needs to provide actual model name
    if model == "custom":
        return [TextContent(type="text", text=json.dumps({
            "step": "model_select",
            "step_number": 1,
            "total_steps": 3,
            "preset": preset,
            "message": "Enter your custom model name.",
            "questions": [
                {
                    "id": "model",
                    "type": "text",
                    "question": "What is the model identifier?",
                    "required": True,
                    "hint": "e.g., gpt-4o-2024-11-20 or anthropic/claude-3-5-sonnet-20241022",
                }
            ],
            "next_action": f"Call config_init(preset='{preset}', model='<actual_model_name>') to continue",
        }, indent=2))]

    # ========================================================================
    # STEP 2: API Key Entry (cloud providers only, when no api_key provided)
    # ========================================================================

    # Determine default API base and key for each preset
    preset_defaults = {
        "local": {
            "api_base": api_base or "http://localhost:1234/v1",
            "api_key": api_key or "lm-studio",
            "requires_api_key": False,
        },
        "openai": {
            "api_base": api_base or "https://api.openai.com/v1",
            "api_key": api_key,  # No default - must be provided
            "requires_api_key": True,
        },
        "anthropic": {
            "api_base": api_base or "https://api.anthropic.com",
            "api_key": api_key,  # No default - must be provided
            "requires_api_key": True,
        },
        "custom": {
            "api_base": api_base,
            "api_key": api_key,
            "requires_api_key": False,  # Optional for custom
        },
    }

    defaults = preset_defaults.get(preset, {})

    # For cloud providers, require API key
    if preset in ["openai", "anthropic"] and not api_key:
        env_var_name = "OPENAI_API_KEY" if preset == "openai" else "ANTHROPIC_API_KEY"
        return [TextContent(type="text", text=json.dumps({
            "step": "api_key_setup",
            "step_number": 2,
            "total_steps": 3,
            "preset": preset,
            "model": model,
            "message": f"{preset.title()} requires an API key. How would you like to configure it?",
            "questions": [
                {
                    "id": "api_key",
                    "type": "select_or_input",
                    "question": "Configure API key:",
                    "options": [
                        {
                            "value": f"${{{env_var_name}}}",
                            "label": "Use environment variable (recommended)",
                            "description": f"References {env_var_name} environment variable",
                            "recommended": True,
                            "setup_instructions": {
                                "windows_powershell": f'$env:{env_var_name} = "your-api-key-here"  # Temporary',
                                "windows_permanent": f'[System.Environment]::SetEnvironmentVariable("{env_var_name}", "your-api-key-here", "User")',
                                "unix": f'export {env_var_name}="your-api-key-here"  # Add to ~/.bashrc for permanent',
                            },
                        },
                        {
                            "value": "paste",
                            "label": "Enter API key directly",
                            "description": "Key will be stored in config file",
                            "warning": "Less secure - key stored in plaintext in config file",
                        },
                    ],
                    "api_key_url": API_KEY_URLS[preset],
                    "api_key_hint": f"Get your API key at {API_KEY_URLS[preset]}",
                }
            ],
            "security_warning": "For production use, we strongly recommend using environment variables instead of storing API keys in config files.",
            "next_action": f"Call config_init(preset='{preset}', model='{model}', api_key='...') to continue",
        }, indent=2))]

    # ========================================================================
    # STEP 3: Connection Validation & Save
    # ========================================================================

    # Build final configuration
    final_model = model
    final_api_base = defaults.get("api_base")
    final_api_key = defaults.get("api_key") or api_key

    # For local preset, apply defaults if not specified
    if preset == "local":
        final_model = model or "openai/qwen/qwen3-coder-30b"
        final_api_base = api_base or "http://localhost:1234/v1"
        final_api_key = api_key or "lm-studio"

    # Validate we have all required fields
    if not final_model:
        return [TextContent(type="text", text=json.dumps({
            "error": "Model is required but not provided",
            "troubleshooting": [
                "Call config_init() to restart the setup wizard",
                f"Or specify model directly: config_init(preset='{preset}', model='your-model')",
            ],
        }, indent=2))]

    if not final_api_base and preset != "custom":
        return [TextContent(type="text", text=json.dumps({
            "error": "API base URL is required but not provided",
            "troubleshooting": [
                "Call config_init() to restart the setup wizard",
                f"Or specify api_base directly: config_init(preset='{preset}', api_base='your-url')",
            ],
        }, indent=2))]

    # Connection validation (unless skipped)
    validation_result = None
    if not skip_validation:
        try:
            # Resolve env vars for validation
            resolved_api_key = final_api_key
            if final_api_key and final_api_key.startswith("${") and final_api_key.endswith("}"):
                env_var = final_api_key[2:-1]
                resolved_api_key = os.environ.get(env_var)
                if not resolved_api_key:
                    # Env var not set - warn but allow saving
                    validation_result = {
                        "status": "warning",
                        "message": f"Environment variable {env_var} is not set. Configuration will be saved but connection cannot be tested.",
                        "troubleshooting": [
                            f"Set the environment variable: {env_var}",
                            "Windows PowerShell: $env:" + env_var + ' = "your-key"',
                            "Unix/Mac: export " + env_var + '="your-key"',
                            "Then restart your terminal/IDE",
                        ],
                    }

            if validation_result is None:
                # Actually test the connection
                provider = create_provider(
                    model=final_model,
                    api_base=final_api_base,
                    api_key=resolved_api_key,
                )

                import time
                start_time = time.time()
                is_valid = provider.test_connection()
                response_time_ms = int((time.time() - start_time) * 1000)

                if is_valid:
                    validation_result = {
                        "status": "success",
                        "message": "Connection successful! Your configuration is working.",
                        "response_time_ms": response_time_ms,
                    }
                else:
                    validation_result = {
                        "status": "failed",
                        "message": "Connection test failed.",
                        "troubleshooting": _get_troubleshooting_tips(preset, final_api_base, final_api_key),
                    }

                    # Return validation failure with retry options
                    return [TextContent(type="text", text=json.dumps({
                        "step": "validate",
                        "step_number": 3,
                        "total_steps": 3,
                        "validation_status": "failed",
                        "message": "Connection failed. Please check your configuration.",
                        "error": validation_result,
                        "retry_options": [
                            {
                                "action": "Change API key",
                                "call": f"config_init(preset='{preset}', model='{final_model}', api_key='<new_key>')",
                            },
                            {
                                "action": "Change provider",
                                "call": "config_init()",
                            },
                            {
                                "action": "Skip validation and save anyway",
                                "call": f"config_init(preset='{preset}', model='{final_model}', api_key='{final_api_key}', skip_validation=True)",
                                "warning": "Not recommended - config may not work",
                            },
                        ],
                    }, indent=2))]

        except Exception as e:
            error_str = str(e)
            validation_result = {
                "status": "failed",
                "message": f"Connection test error: {error_str}",
                "troubleshooting": _get_troubleshooting_tips(preset, final_api_base, final_api_key, error_str),
            }

            return [TextContent(type="text", text=json.dumps({
                "step": "validate",
                "step_number": 3,
                "total_steps": 3,
                "validation_status": "failed",
                "message": "Connection test encountered an error.",
                "error": validation_result,
                "retry_options": [
                    {
                        "action": "Check your settings and try again",
                        "call": f"config_init(preset='{preset}', model='{final_model}', api_key='...')",
                    },
                    {
                        "action": "Skip validation and save anyway",
                        "call": f"config_init(preset='{preset}', model='{final_model}', api_key='{final_api_key}', skip_validation=True)",
                        "warning": "Not recommended - config may not work",
                    },
                ],
            }, indent=2))]

    # ========================================================================
    # Save Configuration
    # ========================================================================
    try:
        config = ConfigSchema(
            defaults=ProviderSettings(
                model=final_model,
                api_base=final_api_base,
                api_key=final_api_key,
                temperature=0.7,
                max_tokens=1024,
            )
        )

        config_path = get_user_config_path()
        save_config(config, config_path)

        # Build success response
        response = {
            "step": "complete",
            "step_number": 3,
            "total_steps": 3,
            "success": True,
            "message": f"Configuration saved successfully with '{preset}' preset!",
            "config_path": str(config_path),
            "summary": {
                "provider": preset.title() if preset != "custom" else "Custom",
                "model": final_model,
                "api_base": final_api_base,
                "api_key": _mask_api_key(final_api_key),
            },
        }

        if validation_result:
            response["validation"] = validation_result

        response["next_steps"] = [
            {
                "action": "Run a test discussion",
                "example": "council_discuss(topic='Test Topic', objective='Verify setup works')",
            },
            {
                "action": "Validate configuration",
                "example": "config_validate()",
            },
            {
                "action": "View full configuration",
                "example": "config_get(resolved=True)",
            },
        ]

        # For backward compatibility, also include the config dump
        response["config"] = config.model_dump(exclude_none=True)
        response["next_step"] = "You can now use council_discuss to run discussions, or use config_validate to test the connection."

        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Error saving configuration: {str(e)}",
            "troubleshooting": [
                "Check write permissions to config directory",
                f"Config path: {get_user_config_path()}",
                "Try running with elevated permissions if needed",
            ],
        }, indent=2))]


def _get_troubleshooting_tips(preset: str, api_base: Optional[str], api_key: Optional[str], error: Optional[str] = None) -> list[str]:
    """Generate troubleshooting tips based on preset and error."""
    tips = []

    if preset == "local":
        tips.extend([
            f"Ensure your local LLM server is running at {api_base}",
            "Check if the server supports OpenAI-compatible API",
            "Try accessing the /models endpoint in a browser",
            "For LM Studio: Enable 'Start Server' in the Local Server tab",
            "For Ollama: Ensure it's running with 'ollama serve'",
        ])
    elif preset == "openai":
        tips.extend([
            "Verify your API key is correct (starts with 'sk-')",
            "Check if the API key has been revoked or expired",
            "Ensure you have billing set up at platform.openai.com",
            "If using env var, restart your terminal after setting it",
        ])
    elif preset == "anthropic":
        tips.extend([
            "Verify your API key is correct",
            "Check if the API key has been revoked or expired",
            "Ensure you have an active account at console.anthropic.com",
            "If using env var, restart your terminal after setting it",
        ])
    else:
        tips.extend([
            f"Check if the API endpoint is accessible: {api_base}",
            "Verify the API key is correct",
            "Ensure the model name is valid for your provider",
        ])

    if error:
        if "timeout" in error.lower():
            tips.insert(0, "Connection timed out - check if the server is running and accessible")
        elif "401" in error or "unauthorized" in error.lower():
            tips.insert(0, "Authentication failed - check your API key")
        elif "404" in error:
            tips.insert(0, "Endpoint not found - verify the API base URL")
        elif "connection refused" in error.lower():
            tips.insert(0, "Connection refused - ensure the server is running")

    return tips


def _mask_api_key(api_key: Optional[str]) -> str:
    """Mask API key for display, keeping env var references visible."""
    if not api_key:
        return "(not set)"
    if api_key.startswith("${") and api_key.endswith("}"):
        return f"{api_key} (from environment)"
    if len(api_key) > 8:
        return f"{api_key[:4]}...{api_key[-4:]}"
    return "****"


async def handle_config_validate(args: dict) -> list[TextContent]:
    """Validate provider configuration."""
    specific_provider = args.get("provider")

    try:
        config = load_config()
        manager = get_config_manager()
        resolved = manager.resolve(config)

        results = {}

        # Validate specific provider or all
        if specific_provider:
            if specific_provider == "default":
                defaults = resolved.defaults
                if defaults.model:
                    try:
                        provider = create_provider(
                            model=defaults.model,
                            api_base=defaults.api_base,
                            api_key=defaults.api_key,
                        )
                        results["default"] = {
                            "valid": provider.test_connection(),
                            "model": defaults.model,
                            "api_base": defaults.api_base,
                        }
                    except Exception as e:
                        results["default"] = {"valid": False, "error": str(e)}
            elif specific_provider in resolved.providers:
                settings = resolved.defaults.merge_with(resolved.providers[specific_provider])
                try:
                    provider = create_provider(
                        model=settings.model,
                        api_base=settings.api_base,
                        api_key=settings.api_key,
                    )
                    results[specific_provider] = {
                        "valid": provider.test_connection(),
                        "model": settings.model,
                        "api_base": settings.api_base,
                    }
                except Exception as e:
                    results[specific_provider] = {"valid": False, "error": str(e)}
            else:
                return [TextContent(type="text", text=f"Provider not found: {specific_provider}")]
        else:
            # Validate all providers
            validation_results = manager.validate_providers(resolved)
            for name, is_valid in validation_results.items():
                if name == "default":
                    results[name] = {
                        "valid": is_valid,
                        "model": resolved.defaults.model,
                        "api_base": resolved.defaults.api_base,
                    }
                elif name in resolved.providers:
                    settings = resolved.providers[name]
                    results[name] = {
                        "valid": is_valid,
                        "model": settings.model,
                        "api_base": settings.api_base,
                    }

        all_valid = all(r.get("valid", False) for r in results.values())

        return [TextContent(type="text", text=json.dumps({
            "all_valid": all_valid,
            "providers": results,
            "config_paths": {
                "user": str(get_user_config_path()),
                "project": str(get_project_config_path()),
            }
        }, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error validating config: {str(e)}")]


async def handle_providers_list(args: dict) -> list[TextContent]:
    """List all configured providers."""
    show_details = args.get("show_details", False)

    try:
        config = load_config()
        manager = get_config_manager()
        resolved = manager.resolve(config)

        providers_info = []

        # Default provider
        defaults = resolved.defaults
        provider_info = {
            "name": "default",
            "model": defaults.model,
            "is_default": True,
        }
        if show_details:
            provider_info["api_base"] = defaults.api_base
            provider_info["temperature"] = defaults.temperature
            provider_info["max_tokens"] = defaults.max_tokens
        providers_info.append(provider_info)

        # Named providers
        for name, settings in resolved.providers.items():
            merged = defaults.merge_with(settings)
            provider_info = {
                "name": name,
                "model": merged.model,
                "is_default": False,
            }
            if show_details:
                provider_info["api_base"] = merged.api_base
                provider_info["temperature"] = merged.temperature
                provider_info["max_tokens"] = merged.max_tokens
                # Show which fields are overridden
                provider_info["overrides"] = [
                    k for k, v in settings.model_dump(exclude_none=True).items()
                    if v is not None
                ]
            providers_info.append(provider_info)

        # Persona-specific configs
        persona_providers = []
        for name, settings in resolved.persona_configs.items():
            persona_providers.append({
                "persona": name,
                "overrides": [k for k, v in settings.model_dump(exclude_none=True).items() if v is not None]
            })

        return [TextContent(type="text", text=json.dumps({
            "providers": providers_info,
            "persona_configs": persona_providers if persona_providers else None,
            "total_providers": len(providers_info),
        }, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error listing providers: {str(e)}")]


async def handle_personas_generate(args: dict) -> list[TextContent]:
    """Generate personas for a topic."""
    topic = args.get("topic")
    count = args.get("count", 3)
    save_to = args.get("save_to")

    if not topic:
        return [TextContent(type="text", text="Error: 'topic' is required")]

    try:
        # Load config for provider settings
        config = load_config()
        manager = get_config_manager()
        resolved = manager.resolve(config)

        # Create provider
        defaults = resolved.defaults
        provider = create_provider(
            model=defaults.model or "openai/qwen/qwen3-coder-30b",
            api_base=defaults.api_base or "http://localhost:1234/v1",
            api_key=defaults.api_key,
        )

        # Generate personas
        persona_manager = PersonaManager(provider=provider)
        personas = persona_manager.generate_personas_for_topic(
            topic=topic,
            count=count,
            save_to=save_to,
        )

        result = {
            "topic": topic,
            "personas": [
                {
                    "name": p.name,
                    "role": p.role,
                    "expertise": p.expertise,
                    "personality_traits": p.personality_traits,
                    "perspective": p.perspective,
                }
                for p in personas
            ],
            "count": len(personas),
        }

        if save_to:
            result["saved_to"] = save_to

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error generating personas: {str(e)}")]


async def _run_server():
    """Run the MCP server (async implementation)."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point for the MCP server.

    Note: Entry points must be synchronous functions.
    This wraps the async server with asyncio.run().
    """
    import asyncio
    asyncio.run(_run_server())


if __name__ == "__main__":
    main()
