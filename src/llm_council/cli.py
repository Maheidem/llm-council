"""Command-line interface for LLM Council."""

import json
import sys
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .models import ConsensusType, DEFAULT_PERSONAS, Persona
from .providers import create_provider, PRESETS
from .personas import PersonaManager
from .council import CouncilEngine


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
    default="openai/qwen3-coder-30b",
    help="Model to use (default: openai/qwen3-coder-30b for LM Studio)"
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
        if "model" in preset_config and model == "openai/qwen3-coder-30b":
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

    if auto_personas:
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
@click.option("--model", "-m", default="openai/qwen3-coder-30b", help="Model to test")
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
        "model": "openai/qwen3-coder-30b",
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
        model=config.get("model", "openai/qwen3-coder-30b"),
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


if __name__ == "__main__":
    main()
