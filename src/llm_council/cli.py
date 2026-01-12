"""Command-line interface for LLM Council."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .models import ConsensusType, DEFAULT_PERSONAS, Persona
from .providers import create_provider, PRESETS, ProviderRegistry
from .personas import PersonaManager
from .council import CouncilEngine
from .config import (
    ConfigManager,
    ConfigSchema,
    ProviderSettings,
    get_user_config_path,
    get_project_config_path,
    load_config,
    save_config,
    get_default_config,
)


console = Console(force_terminal=True, legacy_windows=False)


@click.group()
@click.version_option(version=__version__)
def main():
    """LLM Council - Multi-persona AI deliberation and consensus tool."""
    pass


@main.command()
@click.option("--topic", "-t", required=True, help="Discussion topic")
@click.option("--objective", "-o", required=True, help="Goal/decision to reach")
@click.option("--context", "-c", help="Additional context for the discussion")
@click.option(
    "--model", "-m",
    default="openai/qwen/qwen3-coder-30b",
    help="Model to use (default: openai/qwen/qwen3-coder-30b for LM Studio)"
)
@click.option(
    "--api-base", "-b",
    default="http://localhost:1234/v1",
    help="API base URL (default: http://localhost:1234/v1 for LM Studio)"
)
@click.option("--api-key", "-k", help="API key if required")
@click.option(
    "--preset", "-p",
    type=click.Choice(list(PRESETS.keys())),
    help="Use a preset configuration"
)
@click.option(
    "--personas", "-n",
    default=3,
    type=int,
    help="Number of personas (default: 3)"
)
@click.option(
    "--auto-personas/--default-personas",
    default=False,
    help="Auto-generate personas based on topic"
)
@click.option(
    "--personas-file", "-pf",
    type=click.Path(exists=True),
    help="Load personas from YAML/JSON file (overrides --personas count)"
)
@click.option(
    "--consensus-type",
    type=click.Choice([c.value for c in ConsensusType]),
    default="majority",
    help="Type of consensus required"
)
@click.option(
    "--max-rounds", "-r",
    default=5,
    type=int,
    help="Maximum discussion rounds"
)
@click.option(
    "--output", "-O",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format"
)
@click.option("--quiet", "-q", is_flag=True, help="Minimal output (for automation)")
def discuss(
    topic: str,
    objective: str,
    context: Optional[str],
    model: str,
    api_base: str,
    api_key: Optional[str],
    preset: Optional[str],
    personas: int,
    auto_personas: bool,
    personas_file: Optional[str],
    consensus_type: str,
    max_rounds: int,
    output: str,
    quiet: bool,
):
    """Run a council discussion on a topic.

    Example:
        llm-council discuss -t "API Design" -o "Choose REST vs GraphQL" -n 3
    """
    # Apply preset if specified
    if preset:
        preset_config = PRESETS[preset]
        if "model" in preset_config and model == "openai/qwen/qwen3-coder-30b":
            model = preset_config["model"]
        if "api_base" in preset_config:
            api_base = preset_config["api_base"]
        if "api_key" in preset_config and not api_key:
            api_key = preset_config["api_key"]

    # Create provider
    try:
        provider = create_provider(
            provider_type="litellm",
            model=model,
            api_base=api_base,
            api_key=api_key,
        )
    except Exception as e:
        if not quiet:
            console.print(f"[red]Failed to create provider: {e}[/red]")
        sys.exit(1)

    # Test connection
    if not quiet:
        console.print(f"[dim]Connecting to {api_base}...[/dim]")

    # Create persona manager and get personas
    persona_manager = PersonaManager(provider=provider if auto_personas else None)

    if personas_file:
        # Load from file (highest priority)
        if not quiet:
            console.print(f"[dim]Loading personas from {personas_file}...[/dim]")
        try:
            persona_list = persona_manager.load_personas(personas_file)
        except Exception as e:
            console.print(f"[red]Failed to load personas: {e}[/red]")
            sys.exit(1)
    elif auto_personas:
        if not quiet:
            console.print("[dim]Generating personas for topic...[/dim]")
        persona_list = persona_manager.generate_personas_for_topic(topic, personas)
    else:
        persona_list = persona_manager.get_default_personas(personas)

    if not quiet:
        console.print(f"\n[bold]Council Members:[/bold]")
        for p in persona_list:
            console.print(f"  - {p.name} - {p.role}")

    # Create engine
    engine = CouncilEngine(
        provider=provider,
        consensus_type=ConsensusType(consensus_type),
        max_rounds=max_rounds,
    )

    # Run session
    if not quiet:
        console.print(f"\n[bold cyan]Starting discussion on:[/bold cyan] {topic}")
        console.print(f"[bold cyan]Objective:[/bold cyan] {objective}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=quiet,
    ) as progress:
        task = progress.add_task("Running council session...", total=None)
        session = engine.run_session(
            topic=topic,
            objective=objective,
            personas=persona_list,
            initial_context=context,
        )
        progress.update(task, completed=True)

    # Output results
    if output == "json":
        print(json.dumps(session.to_dict(), indent=2))
    else:
        _print_session_results(session, quiet)


def _print_session_results(session, quiet: bool):
    """Print session results in text format."""
    console.print("\n" + "=" * 60)
    console.print("[bold green]COUNCIL SESSION COMPLETE[/bold green]")
    console.print("=" * 60)

    # Show rounds
    for round_result in session.rounds:
        if not quiet:
            console.print(f"\n[bold]Round {round_result.round_number}:[/bold]")
            for msg in round_result.messages:
                panel = Panel(
                    msg.content,
                    title=f"[cyan]{msg.persona_name}[/cyan]",
                    border_style="dim",
                )
                console.print(panel)

        # Show votes if any
        if round_result.votes:
            console.print("\n[bold]Votes:[/bold]")
            table = Table()
            table.add_column("Persona")
            table.add_column("Vote")
            table.add_column("Reasoning")
            for vote in round_result.votes:
                table.add_row(
                    vote.persona_name,
                    vote.choice.value.upper(),
                    vote.reasoning[:100] + "..." if len(vote.reasoning) > 100 else vote.reasoning,
                )
            console.print(table)

    # Final result
    console.print("\n" + "=" * 60)
    if session.consensus_reached:
        console.print("[bold green][OK] CONSENSUS REACHED[/bold green]")
    else:
        console.print("[bold yellow][!] NO CONSENSUS[/bold yellow]")

    console.print(f"\n[bold]Final Position:[/bold]")
    console.print(Panel(session.final_consensus or "No consensus", border_style="green" if session.consensus_reached else "yellow"))


@main.command()
@click.option(
    "--api-base", "-b",
    default="http://localhost:1234/v1",
    help="API base URL to test"
)
@click.option("--model", "-m", default="openai/qwen/qwen3-coder-30b", help="Model to test")
@click.option("--api-key", "-k", help="API key if required")
def test_connection(api_base: str, model: str, api_key: Optional[str]):
    """Test connection to LLM provider."""
    console.print(f"Testing connection to {api_base}...")

    try:
        provider = create_provider(
            provider_type="litellm",
            model=model,
            api_base=api_base,
            api_key=api_key or "lm-studio",
        )

        if provider.test_connection():
            console.print("[green][OK] Connection successful![/green]")
            sys.exit(0)
        else:
            console.print("[red][FAIL] Connection failed[/red]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[red][ERROR] {e}[/red]")
        sys.exit(1)


@main.command()
def list_personas():
    """List available default personas."""
    console.print("[bold]Default Personas:[/bold]\n")

    for persona in DEFAULT_PERSONAS:
        console.print(Panel(
            f"[bold]{persona.name}[/bold] - {persona.role}\n\n"
            f"[dim]Expertise:[/dim] {', '.join(persona.expertise)}\n"
            f"[dim]Traits:[/dim] {', '.join(persona.personality_traits)}\n"
            f"[dim]Perspective:[/dim] {persona.perspective}",
            border_style="cyan",
        ))


@main.command()
@click.argument("config_file", type=click.Path(exists=True))
def run_config(config_file: str):
    """Run a council session from a JSON config file.

    CONFIG_FILE: Path to JSON configuration file

    Config format:
    {
        "topic": "Discussion topic",
        "objective": "Goal to achieve",
        "context": "Optional context",
        "model": "openai/qwen/qwen3-coder-30b",
        "api_base": "http://localhost:1234/v1",
        "personas": 3,
        "auto_personas": false,
        "consensus_type": "majority",
        "max_rounds": 5,
        "output": "json"
    }
    """
    with open(config_file) as f:
        config = json.load(f)

    # Extract and validate config
    topic = config.get("topic")
    objective = config.get("objective")

    if not topic or not objective:
        console.print("[red]Config must include 'topic' and 'objective'[/red]")
        sys.exit(1)

    # Call discuss with config values
    ctx = click.Context(discuss)
    ctx.invoke(
        discuss,
        topic=topic,
        objective=objective,
        context=config.get("context"),
        model=config.get("model", "openai/qwen/qwen3-coder-30b"),
        api_base=config.get("api_base", "http://localhost:1234/v1"),
        api_key=config.get("api_key"),
        preset=config.get("preset"),
        personas=config.get("personas", 3),
        auto_personas=config.get("auto_personas", False),
        consensus_type=config.get("consensus_type", "majority"),
        max_rounds=config.get("max_rounds", 5),
        output=config.get("output", "text"),
        quiet=config.get("quiet", False),
    )


@main.group()
def config():
    """Manage LLM Council configuration."""
    pass


@config.command("show")
@click.option("--source", "-s", is_flag=True, help="Show source of each value")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def config_show(source: bool, as_json: bool):
    """Show current configuration."""
    manager = ConfigManager()
    cfg = manager.load()

    if as_json:
        print(json.dumps(cfg.model_dump(exclude_none=True), indent=2))
        return

    console.print("[bold]Current Configuration:[/bold]\n")

    # Show config file locations
    user_path = get_user_config_path()
    project_path = get_project_config_path()

    console.print("[dim]Config files:[/dim]")
    console.print(f"  User config:    {user_path} {'[green](exists)[/green]' if user_path.exists() else '[dim](not found)[/dim]'}")
    console.print(f"  Project config: {project_path} {'[green](exists)[/green]' if project_path.exists() else '[dim](not found)[/dim]'}")
    console.print()

    # Show defaults
    console.print("[bold cyan]Defaults:[/bold cyan]")
    defaults = cfg.defaults.model_dump(exclude_none=True)
    if defaults:
        for key, value in defaults.items():
            console.print(f"  {key}: {value}")
    else:
        console.print("  [dim](using built-in defaults)[/dim]")

    # Show generation settings
    console.print("\n[bold cyan]Generation:[/bold cyan]")
    gen = cfg.generation.model_dump(exclude_none=True)
    if gen:
        for key, value in gen.items():
            console.print(f"  {key}: {value}")
    else:
        console.print("  [dim](using defaults)[/dim]")

    # Show named providers
    if cfg.providers:
        console.print("\n[bold cyan]Named Providers:[/bold cyan]")
        for name, settings in cfg.providers.items():
            console.print(f"\n  [yellow]{name}[/yellow]:")
            for key, value in settings.model_dump(exclude_none=True).items():
                # Mask API keys
                if key == 'api_key' and value and not value.startswith('${'):
                    value = value[:8] + "..." if len(value) > 8 else "***"
                console.print(f"    {key}: {value}")

    # Show persona configs
    if cfg.persona_configs:
        console.print("\n[bold cyan]Per-Persona Configs:[/bold cyan]")
        for name, settings in cfg.persona_configs.items():
            console.print(f"\n  [yellow]{name}[/yellow]:")
            for key, value in settings.model_dump(exclude_none=True).items():
                console.print(f"    {key}: {value}")

    # Show council settings
    console.print("\n[bold cyan]Council Settings:[/bold cyan]")
    for key, value in cfg.council.model_dump().items():
        console.print(f"  {key}: {value}")


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--project", "-p", is_flag=True, help="Set in project config instead of user config")
def config_set(key: str, value: str, project: bool):
    """Set a configuration value.

    Examples:
        llm-council config set defaults.model gpt-4o
        llm-council config set defaults.temperature 0.8
        llm-council config set defaults.api_base http://localhost:1234/v1
    """
    path = get_project_config_path() if project else get_user_config_path()

    # Load existing config or create new
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {'version': '1.0'}

    # Parse the key path (e.g., "defaults.model")
    parts = key.split('.')

    # Convert value type if needed
    if value.lower() in ('true', 'false'):
        value = value.lower() == 'true'
    elif value.isdigit():
        value = int(value)
    else:
        try:
            value = float(value)
        except ValueError:
            pass  # Keep as string

    # Navigate/create the path and set value
    current = data
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value

    # Save
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Set {key} = {value}[/green]")
    console.print(f"[dim]Saved to {path}[/dim]")


@config.command("unset")
@click.argument("key")
@click.option("--project", "-p", is_flag=True, help="Unset in project config")
def config_unset(key: str, project: bool):
    """Remove a configuration value."""
    path = get_project_config_path() if project else get_user_config_path()

    if not path.exists():
        console.print(f"[yellow]Config file not found: {path}[/yellow]")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    # Navigate to parent and delete key
    parts = key.split('.')
    current = data
    for part in parts[:-1]:
        if part not in current:
            console.print(f"[yellow]Key not found: {key}[/yellow]")
            return
        current = current[part]

    if parts[-1] in current:
        del current[parts[-1]]
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        console.print(f"[green]Removed {key}[/green]")
    else:
        console.print(f"[yellow]Key not found: {key}[/yellow]")


@config.command("reset")
@click.option("--project", "-p", is_flag=True, help="Reset project config")
@click.confirmation_option(prompt="Are you sure you want to reset configuration?")
def config_reset(project: bool):
    """Reset configuration to defaults."""
    path = get_project_config_path() if project else get_user_config_path()

    if path.exists():
        path.unlink()
        console.print(f"[green]Removed {path}[/green]")
    else:
        console.print(f"[yellow]No config file found at {path}[/yellow]")


@config.command("init")
@click.option("--project", "-p", is_flag=True, help="Initialize project config")
def config_init(project: bool):
    """Initialize configuration interactively."""
    path = get_project_config_path() if project else get_user_config_path()

    if path.exists():
        if not click.confirm(f"Config already exists at {path}. Overwrite?"):
            return

    console.print("[bold]LLM Council Configuration Setup[/bold]\n")

    # Get model
    model = click.prompt(
        "Default model",
        default="openai/qwen/qwen3-coder-30b",
    )

    # Get API base
    api_base = click.prompt(
        "API base URL",
        default="http://localhost:1234/v1",
    )

    # Get API key
    api_key = click.prompt(
        "API key (use ${ENV_VAR} for env variable)",
        default="${OPENAI_API_KEY}",
    )

    # Get temperature
    temperature = click.prompt(
        "Default temperature",
        default=0.7,
        type=float,
    )

    # Get max tokens
    max_tokens = click.prompt(
        "Default max tokens",
        default=1024,
        type=int,
    )

    # Create config
    data = {
        'version': '1.0',
        'defaults': {
            'model': model,
            'api_base': api_base,
            'api_key': api_key,
            'temperature': temperature,
            'max_tokens': max_tokens,
        },
    }

    # Save
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]Configuration saved to {path}[/green]")
    console.print("\n[dim]You can edit this file directly or use 'llm-council config set' to modify values.[/dim]")


@config.command("export")
@click.argument("output_file", type=click.Path())
def config_export(output_file: str):
    """Export configuration to a file."""
    manager = ConfigManager()
    cfg = manager.load()

    output_path = Path(output_file)
    data = cfg.model_dump(exclude_none=True)

    if output_path.suffix.lower() in ('.yaml', '.yml'):
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    console.print(f"[green]Configuration exported to {output_path}[/green]")


@config.command("import")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--project", "-p", is_flag=True, help="Import as project config")
def config_import(input_file: str, project: bool):
    """Import configuration from a file."""
    input_path = Path(input_file)
    output_path = get_project_config_path() if project else get_user_config_path()

    # Load input file
    if input_path.suffix.lower() in ('.yaml', '.yml'):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # Validate
    try:
        ConfigSchema(**data)
    except Exception as e:
        console.print(f"[red]Invalid configuration: {e}[/red]")
        sys.exit(1)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Configuration imported to {output_path}[/green]")


@config.command("validate")
def config_validate():
    """Validate current configuration and test providers."""
    manager = ConfigManager()

    try:
        cfg = manager.load()
        console.print("[green][OK] Configuration is valid[/green]\n")
    except Exception as e:
        console.print(f"[red][FAIL] Configuration invalid: {e}[/red]")
        sys.exit(1)

    # Resolve config
    resolved = manager.resolve(cfg)

    # Test providers
    console.print("[bold]Testing providers...[/bold]\n")

    # Test default
    if resolved.defaults.model:
        console.print(f"  default ({resolved.defaults.model})...", end=" ")
        try:
            provider = create_provider(
                model=resolved.defaults.model,
                api_base=resolved.defaults.api_base,
                api_key=resolved.defaults.api_key,
                temperature=resolved.defaults.temperature or 0.7,
                max_tokens=resolved.defaults.max_tokens or 1024,
            )
            if provider.test_connection():
                console.print("[green]OK[/green]")
            else:
                console.print("[red]FAILED[/red]")
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")

    # Test named providers
    for name, settings in resolved.providers.items():
        merged = resolved.defaults.merge_with(settings)
        console.print(f"  {name} ({merged.model})...", end=" ")
        try:
            provider = create_provider(
                model=merged.model or "openai/qwen/qwen3-coder-30b",
                api_base=merged.api_base,
                api_key=merged.api_key,
                temperature=merged.temperature or 0.7,
                max_tokens=merged.max_tokens or 1024,
            )
            if provider.test_connection():
                console.print("[green]OK[/green]")
            else:
                console.print("[red]FAILED[/red]")
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")


@main.group()
def providers():
    """Manage LLM providers."""
    pass


@providers.command("list")
def providers_list():
    """List available providers."""
    console.print("[bold]Built-in Presets:[/bold]\n")

    table = Table()
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("API Base")

    for name, preset in PRESETS.items():
        table.add_row(
            name,
            preset.get("model", "(default)"),
            preset.get("api_base", "(default)"),
        )

    console.print(table)

    # Show configured providers
    manager = ConfigManager()
    try:
        cfg = manager.load()
        if cfg.providers:
            console.print("\n[bold]Configured Providers:[/bold]\n")
            table = Table()
            table.add_column("Name")
            table.add_column("Model")
            table.add_column("API Base")

            for name, settings in cfg.providers.items():
                table.add_row(
                    name,
                    settings.model or "(inherit)",
                    settings.api_base or "(inherit)",
                )

            console.print(table)
    except Exception:
        pass


@providers.command("test")
@click.argument("name", required=False)
def providers_test(name: Optional[str]):
    """Test provider connection."""
    manager = ConfigManager()
    cfg = manager.load()
    resolved = manager.resolve(cfg)

    if name:
        # Test specific provider
        if name in resolved.providers:
            settings = resolved.defaults.merge_with(resolved.providers[name])
        elif name == 'default':
            settings = resolved.defaults
        elif name in PRESETS:
            preset = PRESETS[name]
            settings = ProviderSettings(
                model=preset.get("model"),
                api_base=preset.get("api_base"),
                api_key=preset.get("api_key"),
            )
            settings = resolved.defaults.merge_with(settings)
        else:
            console.print(f"[red]Provider not found: {name}[/red]")
            sys.exit(1)

        console.print(f"Testing {name}...", end=" ")
        try:
            provider = create_provider(
                model=settings.model or "openai/qwen/qwen3-coder-30b",
                api_base=settings.api_base,
                api_key=settings.api_key,
                temperature=settings.temperature or 0.7,
                max_tokens=settings.max_tokens or 1024,
            )
            if provider.test_connection():
                console.print("[green]OK[/green]")
            else:
                console.print("[red]FAILED[/red]")
                sys.exit(1)
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
            sys.exit(1)
    else:
        # Test all configured providers
        ctx = click.Context(config_validate)
        ctx.invoke(config_validate)


if __name__ == "__main__":
    main()
