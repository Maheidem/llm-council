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
            description="Initialize LLM Council configuration interactively. Use this for first-time setup or to reconfigure. Returns a form structure the agent can use to gather required information from the user.",
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
                        "description": "Model name/identifier to use"
                    },
                    "api_base": {
                        "type": "string",
                        "description": "API base URL (required for local/custom)"
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key (use ${ENV_VAR} syntax for security)"
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

        # Get personas
        persona_manager = PersonaManager(provider=provider)
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
    """Initialize configuration - onboarding flow."""
    preset = args.get("preset")
    model = args.get("model")
    api_base = args.get("api_base")
    api_key = args.get("api_key")

    # If no preset provided, return the setup form/guidance
    if not preset:
        return [TextContent(type="text", text=json.dumps({
            "action": "onboarding_required",
            "message": "LLM Council configuration setup. Please help the user choose a configuration.",
            "questions": [
                {
                    "id": "preset",
                    "question": "Which LLM provider would you like to use?",
                    "options": [
                        {"value": "local", "label": "Local (LM Studio, Ollama, etc.)", "description": "Run models locally via OpenAI-compatible API"},
                        {"value": "openai", "label": "OpenAI", "description": "Use OpenAI's API (requires API key)"},
                        {"value": "anthropic", "label": "Anthropic Claude", "description": "Use Anthropic's Claude API (requires API key)"},
                        {"value": "custom", "label": "Custom Provider", "description": "Configure a custom OpenAI-compatible endpoint"},
                    ]
                }
            ],
            "next_step": "After user selects preset, call config_init with preset parameter to get next questions."
        }, indent=2))]

    # Generate config based on preset
    presets = {
        "local": {
            "model": model or "openai/qwen/qwen3-coder-30b",
            "api_base": api_base or "http://localhost:1234/v1",
            "api_key": api_key or "lm-studio",
            "questions_if_missing": [
                {"id": "api_base", "question": "What is your local LLM server URL?", "default": "http://localhost:1234/v1"},
                {"id": "model", "question": "Which model are you using?", "default": "openai/qwen/qwen3-coder-30b"},
            ]
        },
        "openai": {
            "model": model or "gpt-4o",
            "api_base": api_base or "https://api.openai.com/v1",
            "api_key": api_key or "${OPENAI_API_KEY}",
            "questions_if_missing": [
                {"id": "api_key", "question": "What is your OpenAI API key? (Use ${OPENAI_API_KEY} for env var)", "sensitive": True},
                {"id": "model", "question": "Which model do you want to use?", "default": "gpt-4o"},
            ]
        },
        "anthropic": {
            "model": model or "anthropic/claude-3-5-sonnet-20241022",
            "api_base": api_base or "https://api.anthropic.com",
            "api_key": api_key or "${ANTHROPIC_API_KEY}",
            "questions_if_missing": [
                {"id": "api_key", "question": "What is your Anthropic API key? (Use ${ANTHROPIC_API_KEY} for env var)", "sensitive": True},
                {"id": "model", "question": "Which model do you want to use?", "default": "anthropic/claude-3-5-sonnet-20241022"},
            ]
        },
        "custom": {
            "model": model,
            "api_base": api_base,
            "api_key": api_key,
            "questions_if_missing": [
                {"id": "api_base", "question": "What is your API endpoint URL?", "required": True},
                {"id": "model", "question": "What is the model identifier?", "required": True},
                {"id": "api_key", "question": "What is your API key? (Use ${ENV_VAR} syntax for security)", "sensitive": True},
            ]
        }
    }

    preset_config = presets.get(preset)
    if not preset_config:
        return [TextContent(type="text", text=f"Unknown preset: {preset}")]

    # Check if we have all required info
    missing_questions = []
    for q in preset_config.get("questions_if_missing", []):
        field_id = q["id"]
        if not args.get(field_id) and not preset_config.get(field_id):
            if q.get("required", False) or preset == "custom":
                missing_questions.append(q)

    if missing_questions and preset == "custom":
        return [TextContent(type="text", text=json.dumps({
            "action": "need_more_info",
            "preset": preset,
            "questions": missing_questions,
            "message": "Please gather the following information from the user:"
        }, indent=2))]

    # Create and save configuration
    try:
        config = ConfigSchema(
            defaults=ProviderSettings(
                model=preset_config["model"],
                api_base=preset_config["api_base"],
                api_key=preset_config["api_key"],
                temperature=0.7,
                max_tokens=1024,
            )
        )

        config_path = get_user_config_path()
        save_config(config, config_path)

        return [TextContent(type="text", text=json.dumps({
            "success": True,
            "message": f"Configuration initialized with '{preset}' preset",
            "config_path": str(config_path),
            "config": config.model_dump(exclude_none=True),
            "next_step": "You can now use council_discuss to run discussions, or use config_validate to test the connection."
        }, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error initializing config: {str(e)}")]


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


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
